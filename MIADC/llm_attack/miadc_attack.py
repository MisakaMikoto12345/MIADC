import torch
import torch.nn.functional as F

from .tools import check_legal_input, get_embedding_matrix, HighDimFuzzyAttention
from utils.generate import EmulatorGenerator

def get_illegal_tokens(tokenizer):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    ascii_toks = tuple(set(ascii_toks))
    return ascii_toks


class MIADCAttack:
    def __init__(self,
                 model,
                 tokenizer=None,
                 num_starts=1,
                 num_steps=2000,
                 learning_rate=10,
                 momentum=0.99,
                 use_kv_cache=True,
                 judger=None,
                 ref_base_model = None,
                 ref_finetune_model=None
                ):

        self.model = model
        self.ref_base_model = ref_base_model
        self.ref_finetune_model = ref_finetune_model
        self.tokenizer = tokenizer
        self.num_starts = num_starts
        self.num_steps = num_steps

        self.lr = learning_rate
        self.momentum = momentum

        self.device = model.device
        self.dtype = model.dtype
        self.use_kv_cache = use_kv_cache

        embed_mat = get_embedding_matrix(model)
        self.embed_mat = embed_mat.float()
        self.vocal_size = embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.buffer_size = 64

        # sparsity setting
        self.illegal_tokens = get_illegal_tokens(tokenizer)


        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.top_p = None
        gen_config.temperature = None
        self.gen_config = gen_config
        self.judger = judger

        self.fuzzy_attn = HighDimFuzzyAttention(hidden_dim=self.vocal_size)
        # self.fuzzy_attn.to(self.device)

        self.generator = EmulatorGenerator(self.model, self.tokenizer)


    # def get_optimizer(self, num_adv_tokens, init_opt):
    #     if init_opt is not None:
    #         soft_opt = init_opt
    #     elif init_opt is None:
    #         soft_opt = torch.randn(self.num_starts,
    #                                num_adv_tokens,
    #                                self.vocal_size)
    #     soft_opt[..., self.illegal_tokens] = -10**10
    #     soft_opt = soft_opt.softmax(dim=2)
    #
    #     soft_opt = soft_opt.to(self.device)
    #     # soft_opt.requires_grad = True
    #     soft_opt = soft_opt.clone().detach().requires_grad_(True)
    #     # soft_opt = soft_opt.clone().requires_grad_(True)
    #
    #     lr = self.lr * self.num_starts
    #     optimizer = torch.optim.SGD([soft_opt], lr=lr, momentum=self.momentum)
    #     return soft_opt, optimizer

    def get_optimizer(self, num_adv_tokens, init_opt, step_before):
        # ³õÊ¼»¯£¨±£³ÖÔ­Ê¼Âß¼­£©
        if init_opt is not None:
            soft_opt = init_opt.to(self.device)
        else:
            soft_opt = torch.randn(self.num_starts, num_adv_tokens, self.vocal_size,device=self.device)


        # Ä£ºý»¯´¦Àí£¨ÏÈÓÚ·Ç·¨token´¦Àí£©
        if hasattr(self, 'fuzzy_attn'):
            self.fuzzy_attn = self.fuzzy_attn.to(self.device)
            soft_opt = self.fuzzy_attn(soft_opt,step_before,self.num_steps)  # Ó¦ÓÃÄ£ºý×¢ÒâÁ¦

        # ·Ç·¨token´¦Àí£¨ºóÖÃ¹Ø¼ü²½Öè£©
        soft_opt[..., self.illegal_tokens] = -10 ** 10
        soft_opt = soft_opt.softmax(dim=2)

        # ÌÝ¶ÈÉèÖÃ
        soft_opt = soft_opt.clone().detach().requires_grad_(True)

        # ÓÅ»¯Æ÷ÅäÖÃ
        params = [{'params': [soft_opt], 'lr': self.lr * self.num_starts}]
        if hasattr(self, 'fuzzy_attn'):
            params.append({'params': self.fuzzy_attn.parameters(), 'lr': self.lr * 0.5})

        optimizer = torch.optim.SGD(params, momentum=self.momentum)
        return soft_opt, optimizer

    def contains_restricted_words(self,sentence):
        restricted_words = {"cheat", "bomb", "explosive", "suicide","fraud","harassment","crime"}
        return any(word in sentence.lower() for word in restricted_words)


    def to_recoverable(self, x):
        gen_str = self.tokenizer.decode(x)
        y = self.tokenizer.encode(gen_str, add_special_tokens=False)
        return tuple(y)

    @torch.no_grad()
    def make_sparse(self, soft_opt, all_sparsity):
        point = soft_opt.detach().clone()

        sparsity = all_sparsity.int().view(-1, 1)
        sparsity = sparsity.expand(-1, self.num_adv_tokens).clone()
        s_floor = (all_sparsity % 1 * self.num_adv_tokens).int()
        s_floor = s_floor.clamp(min=5)
        for idx in range(self.num_starts):
            sparsity[idx, :s_floor[idx]] += 1

        sparsity = sparsity[:, torch.randperm(self.num_adv_tokens)]

        mask = torch.zeros_like(soft_opt, dtype=torch.bool)
        for i in range(self.num_starts):
            for j in range(self.num_adv_tokens):
                s = sparsity[i, j].item()
                top_s = point[i, j].topk(k=s)[1]
                mask[i, j, top_s] = 1

        point = torch.where(mask, point.relu() + 1e-6, 0)
        point_grad = torch.where(mask, soft_opt.grad, 0)
        point /= point.sum(dim=2, keepdim=True)
        return point,point_grad

    @torch.no_grad()

    def evaluate(self, buffer_set, gt_label, iter_step):

        adv_tokens = list(buffer_set)


        if len(adv_tokens) < self.buffer_size:
            adv_tokens += adv_tokens[:1] * (self.buffer_size - len(adv_tokens))


        adv_tokens_tensor = torch.tensor([token[0] for token in adv_tokens], dtype=torch.int64, device=self.device)

        if self.use_kv_cache:
            full_samples = torch.cat([adv_tokens_tensor, self.right_ids], dim=1)


            prefix_cache = self.get_cache(batch_size=full_samples.shape[0])
            outputs = self.model(input_ids=full_samples, past_key_values=prefix_cache)

        else:
            full_samples = torch.cat([self.left_ids, adv_tokens_tensor, self.right_ids], dim=1)
            outputs = self.model(input_ids=full_samples)

        outputs = outputs.logits[:, self.logit_slice]
        pred = outputs.argmax(dim=-1)

        accuracies = pred.eq(gt_label).float().mean(1)
        best_acc = accuracies.max().item()

        losses = self.loss_fn(outputs.mT, gt_label)
        losses = losses.mean(1)
        best_loss = losses.min().item()

        # ?¨°¦Ì??e¨º¡ì¡Á?D?¦Ì? adv_token
        best_adv = adv_tokens[losses.argmin().item()]
        best_adv = torch.tensor(best_adv[0], dtype=torch.int64, device=self.device)

        acc_damand = 1
        # if iter_step >= 2500:
        #     acc_damand = best_acc

        if best_acc == 1 or iter_step == self.num_steps-1:
            # idxes = torch.where(accuracies == 1)[0][:2]
            if best_acc == 1 :
                idxes = torch.where(accuracies == 1 )[0][:2]
            elif iter_step == self.num_steps-1 :
                idxes = torch.where(accuracies >= best_acc)[0][:2]



        # if best_acc >= acc_damand:

            # valid_accuracies = accuracies[accuracies == acc_damand]
            #
            # idxes = torch.where(accuracies == acc_damand)[0][torch.topk(accuracies[accuracies == acc_damand], k=min(2, len(valid_accuracies))).indices] # ¨¨??¡ã¨¢???¡Á?¨¨¡¤?¨º?a1¦Ì??¨´¡À?
            gen_strings = []
            flag = 0
            for idx in idxes:
                good_sample = adv_tokens[idx.item()] # ??¨¨? (adv_token, label) ?a¡Á¨¦
                good_adv = torch.tensor(good_sample[0], dtype=torch.int64, device=self.device)
                gen_string,flag = self.further_check(good_adv,flag,best_acc)
                gen_strings.append(gen_string)
                if flag == len(idxes):# ?a¨¤?¦Ì? good_sample[0] ¨º? adv_token
                    print(good_sample[1])

                    return best_acc, best_loss, good_adv, True ,gen_strings


        return best_acc, best_loss, best_adv, False, None



    @torch.no_grad()
    def further_check(self, good_sample,flag,best_acc):

        if best_acc >= 0.01 :
        # if best_acc > 0:
            good_sample_suffix = good_sample.view(1, -1)
        # good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]],
        #                         dim=1)
        # good_sample = torch.cat([self.left_user[:1],good_sample,self.right_user[:1]],
        #                         dim=1)
        # good_sample2 = torch.cat([self.left_user[:1],self.right_user[:1]],
        #                         dim=1)
            # use
            # good_sample = torch.cat([self.right_user[:1],good_sample_suffix,self.right_ids[:1]],
            #                         dim=1)
            #for r2d2
            # good_sample = self.right_ids[:1]
            good_sample = self.right_user[:1]

            # good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nMalicious Assistant:"

            #for r2d2
            good_sample_string = self.tokenizer.decode(self.left_user[:1][0][-9:])+ self.tokenizer.decode(good_sample[0])+ self.tokenizer.decode(self.right_ids[:1][0][:10])


            # if self.contains_restricted_words(good_sample_string):
            if False:
                good_sample = self.right_user[:1]
                #suicide
                # good_sample_string = "User: "+self.tokenizer.decode(good_sample[0])+" (suicide Assistant output is harmful)\nsuicide Assistant:"
                #
                # good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nHarmful suicide Assistant:"

                # good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " (your output is harmful)\nYour output:"
                good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nAssistant: "
                good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))
                full_samples = [[1] + good_sample.cpu().numpy().tolist()]

                use_small_model = True
            else:
                good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))


                full_samples = [good_sample.cpu().numpy().tolist()]

                use_small_model = False

        else:
            good_sample = self.right_user[:1]

            good_sample_string = "User: "+self.tokenizer.decode(good_sample[0])+" \nAssistant:"
            good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))
            full_samples = [[1] + good_sample.cpu().numpy().tolist()]

            use_small_model= True
        # good_sample = self.right_ids[:1]



        # good_sample = good_sample[:, self.u_prompt_start:]
        # good_sample2 = good_sample2[:, :self.target_start]
        # full_samples = good_sample.cpu().numpy().tolist()
        # full_samples2 = good_sample2.cpu().numpy().tolist()
        output = self.generator.generate_with_ref(ref_base_model=self.ref_base_model,
                                                   ref_finetune_model=self.ref_finetune_model,
                                                   prompt_tokens = full_samples,
                                                   max_gen_len = 156,
                                                   temperature=0.7,
                                                   top_p=0.9,
                                                   beta=1.0,
                                                   target_start = self.target_start,
                                                   u_prompt_start = self.u_prompt_start,
                                                   best_acc = best_acc,
                                                   use_small_model = use_small_model
                                                   )

        # output2 = self.generator.generate_with_ref(ref_base_model=self.ref_base_model,
        #                                            ref_finetune_model=self.ref_finetune_model,
        #                                            prompt_tokens = full_samples2,
        #                                            max_gen_len = 256,
        #                                            temperature=0.1,
        #                                            top_p=0.9,
        #                                            beta=1.0,
        #                                            )


        # output = self.model.generate(input_ids=good_sample,
        #                              generation_config=self.gen_config,
        #                              max_new_tokens=512)
        gen_str = self.tokenizer.decode(output.reshape(-1))
        return gen_str , flag+1

        if self.judger is not None:
            return self.judger(self.user_prompt, gen_str)
        else:
            return self.response in gen_str

    @torch.no_grad()
    def get_cache(self, batch_size):
        assert self.use_kv_cache
        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids[:1], use_cache=True)
            self.prefix_cache = outputs.past_key_values

        if batch_size == 1:
            prefix_cache = self.prefix_cache
        else:
            prefix_cache = [(i.expand(batch_size, -1, -1, -1),
                             j.expand(batch_size, -1, -1, -1))
                            for i, j in self.prefix_cache]
        return prefix_cache

        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids, use_cache=True)
            self.prefix_cache = outputs.past_key_values

        if batch_size <= self.buffer_size:
            prefix_cache = [(i[:batch_size], j[:batch_size])
                            for i, j in self.prefix_cache]
        else:
            prefix_cache = [(torch.tile(i[:1], dims=[batch_size, 1, 1, 1]),
                             torch.tile(j[:1], dims=[batch_size, 1, 1, 1]))
                            for i, j in self.prefix_cache]
        return prefix_cache

    def clean_cache(self):
        self.num_adv_tokens = None
        self.left_ids = None
        self.right_ids = None
        self.logit_slice = None
        self.target_start = None
        self.request = None
        self.response = None
        if self.use_kv_cache:
            self.prefix_cache = None
        torch.cuda.empty_cache()

    def attack(self, tokens, slices, user_prompt=None, response=None, init_opt=None,step_before=None):
        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.view(1, -1).to(self.device)
        check_legal_input(tokens, slices)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop
        self.num_adv_tokens = adv_stop - adv_start

        user_prompt_start = slices['user_prompt_slice'].start
        user_prompt_stop = slices['user_prompt_slice'].stop

        soft_opt, optimizer = self.get_optimizer(self.num_adv_tokens,init_opt,step_before)


        # prepare some stuffs
        embeds = self.model.model.embed_tokens(tokens).detach()
        left = embeds[:, :adv_start].expand(self.num_starts, -1, -1)
        right = embeds[:, adv_stop:].expand(self.num_starts, -1, -1)

        self.left_ids = tokens[:, :adv_start].expand(self.buffer_size, -1)
        self.right_ids = tokens[:, adv_stop:].expand(self.buffer_size, -1)

        self.left_user = tokens[:, :user_prompt_start].expand(self.buffer_size, -1)
        self.right_user = tokens[:, user_prompt_start:adv_start].expand(self.buffer_size, -1)

        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        self.target_start = target_start

        self.u_prompt_start = slices['user_prompt_slice'].start

        gt_label = tokens[:, target_start:target_stop]
        gt_label = gt_label.expand(self.buffer_size, -1)

        self.logit_slice = slice(target_start - 1, target_stop - 1)
        if self.use_kv_cache:
            self.logit_slice = slice(target_start - 1 - adv_start,
                                     target_stop - 1 - adv_start)
        # prepare some stuffs end

        seen_set, buffer_set = set(), set()
        onehot_loss, onehot_acc = 1000, 0
        final_adv = tokens[0, slices['adv_slice']]

        for step_ in range(self.num_steps):
            optimizer.zero_grad()

            adv_embeds = (soft_opt @ self.embed_mat).to(self.dtype)
            if self.use_kv_cache:
                full_embeds = torch.cat([adv_embeds, right], dim=1)
                prefix_cache = self.get_cache(batch_size=adv_embeds.shape[0])
                outputs = self.model(inputs_embeds=full_embeds,
                                     past_key_values=prefix_cache)
            else:
                full_embeds = torch.cat([left, adv_embeds, right], dim=1)
                outputs = self.model(inputs_embeds=full_embeds)

            logits = outputs.logits[:, self.logit_slice]

            loss_per_sample = self.loss_fn(logits.mT,
                                           gt_label[:self.num_starts])

            # loss_per_sample = loss_per_sample.mean(1, keepdim=True)

            ell = loss_per_sample.mean()
            ell.backward()

            # soft_opt_before = soft_opt.clone()

            optimizer.step()

            # soft_opt_after = soft_opt

            # Save the next sentence's initialization vector.
            init_soft_opt = soft_opt.clone()


            wrong_pred = logits.argmax(dim=2) != gt_label[:self.num_starts]
            wrong_count = wrong_pred.float().sum(1)

            if step_ == 0:
                running_wrong = wrong_count
            else:
                running_wrong += (wrong_count - running_wrong) * 0.01

            sparsity = (2 ** running_wrong).clamp(max=self.vocal_size / 2)

            soft_opt.data[..., self.illegal_tokens] = -1000
            # last_soft_opt = soft_opt.detach().clone()

            sparse_soft_opt , soft_opt_grad = self.make_sparse(soft_opt, sparsity)
            soft_opt.data.copy_(sparse_soft_opt)
            last_soft_opt = sparse_soft_opt.detach().clone()

            # one hot evaluation

            adv_tokens = []
            for one_soft_opt,one_opt_grad in zip(last_soft_opt,sparse_soft_opt):
                adv_token = one_soft_opt.argmax(dim=1)
                rows = torch.arange(adv_token.size(0), device=adv_token.device)
                adv_grad = one_opt_grad[rows, adv_token]

                adv_token = tuple(adv_token.tolist())
                adv_token1 = self.to_recoverable(adv_token)
                if adv_token1 not in seen_set and len(adv_token1) == self.num_adv_tokens:
                    adv_tokens.append((adv_token1,'original'))
                    seen_set.add(adv_token1)
                    continue

                K = len(adv_token1) - self.num_adv_tokens
                if K > 0:
                    abs_grad = torch.abs(adv_grad)
                    _, topk_indices = torch.topk(abs_grad, k=K)

                    new_adv_token = list(adv_token)
                    for idx in sorted(topk_indices, reverse=True):
                        del new_adv_token[idx]

                    new_adv_token1 = self.to_recoverable(tuple(new_adv_token))
                    if len(new_adv_token1) < self.num_adv_tokens:
                        pad_num = self.num_adv_tokens - len(new_adv_token1)
                        new_adv_token1 = list(new_adv_token1)
                        if self.tokenizer.pad_token_id is not None:
                            new_adv_token1 += [self.tokenizer.pad_token_id] * pad_num
                        else:
                            new_adv_token1 += [self.tokenizer.unk_token_id] * pad_num
                        new_adv_token1 = tuple(new_adv_token1)

                    if new_adv_token1 not in seen_set and len(new_adv_token1) == self.num_adv_tokens:
                        adv_tokens.append((new_adv_token1,'adjusted'))
                        seen_set.add(new_adv_token1)

                if K < 0:
                    new_adv_token = list(adv_token1)
                    if self.tokenizer.pad_token_id is not None:
                        new_adv_token += [self.tokenizer.pad_token_id] * abs(K)
                    else:
                        new_adv_token += [self.tokenizer.unk_token_id] * abs(K)
                    new_adv_token = tuple(new_adv_token)

                    if new_adv_token not in seen_set and len(new_adv_token) == self.num_adv_tokens:
                        adv_tokens.append((new_adv_token,'adjusted'))
                        seen_set.add(new_adv_token)
            del soft_opt_grad


            for adv_token in adv_tokens:
                buffer_set.add(adv_token)
                if len(buffer_set) == self.buffer_size:
                    out = self.evaluate(buffer_set, gt_label,step_)
                    batch_acc, batch_loss, best_adv, early_stop ,gen_string= out

                    onehot_acc = max(onehot_acc, batch_acc)
                    if batch_loss < onehot_loss:
                        onehot_loss = batch_loss
                        final_adv = best_adv

                    print(f'iter:{step_}, '
                          f'loss_batch:{ell: .2f}, '
                          f'best_loss:{onehot_loss: .2f}, '
                          f'best_acc:{onehot_acc: .2f}')

                    if early_stop:
                        print('Early Stop with an Exact Match!')
                        self.clean_cache()
                        return onehot_loss, best_adv.cpu(), step_, init_soft_opt ,gen_string
                    buffer_set = set()

        if len(buffer_set) > 0:
            out = self.evaluate(buffer_set, gt_label, step_)
            batch_acc, batch_loss, best_adv, early_stop ,gen_string= out

            onehot_acc = max(onehot_acc, batch_acc)
            if batch_loss < onehot_loss:
                onehot_loss = batch_loss
                final_adv = best_adv

            print(f'iter:{step_}, '
                  f'loss_batch:{ell: .2f}, '
                  f'best_loss:{onehot_loss: .2f}, '
                  f'best_acc:{onehot_acc: .2f}')

            if early_stop:
                print('Early Stop with an Exact Match!')
                self.clean_cache()
                return onehot_loss, best_adv.cpu(), step_, init_soft_opt, gen_string

        self.clean_cache()
        return onehot_loss, final_adv.cpu(), step_ + 1, init_soft_opt, gen_string


