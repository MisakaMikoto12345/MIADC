import torch
import torch.nn.functional as F

from .tools import check_legal_input, get_embedding_matrix
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


class GCGAttack:
    def __init__(self,
                 model,
                 tokenizer,
                 num_steps=500,
                 topK=256,
                 batch_size=512,
                 use_kv_cache=True,
                 judger=None,
                 ref_base_model=None,
                 ref_finetune_model=None
                 ):
        self.model = model
        self.ref_base_model = ref_base_model
        self.ref_finetune_model = ref_finetune_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps

        self.topK = topK
        self.bs = batch_size

        self.device = model.device
        self.dtype = model.dtype
        self.use_kv_cache = use_kv_cache

        embed_mat = get_embedding_matrix(model)
        self.embed_mat = embed_mat.float()
        self.vocal_size = embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.illegal_tokens = get_illegal_tokens(tokenizer)

        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.top_p = None
        gen_config.temperature = None
        self.gen_config = gen_config
        self.judger = judger

        self.generator = EmulatorGenerator(self.model, self.tokenizer)


    def contains_restricted_words(self,sentence):
        restricted_words = {"cheat", "bomb", "explosive", "suicide","fraud","harassment","crime"}
        return any(word in sentence.lower() for word in restricted_words)

    @torch.no_grad()
    def evaluate(self, resamples, gt_label, step):
        bs = resamples.shape[0]
        buffer_size = 64
        rounds = bs // buffer_size + (bs % buffer_size != 0)
        if self.use_kv_cache:
            full_samples = torch.cat([resamples, self.right_ids], dim=1)
            outputs = []
            for k in range(rounds):
                part_samples = full_samples[k*buffer_size: (k+1)*buffer_size]
                batch_size = part_samples.shape[0]
                prefix_cache = self.get_cache(batch_size=batch_size)
                part_outputs = self.model(input_ids=part_samples,
                                          past_key_values=prefix_cache)
                outputs.append(part_outputs.logits)
                del part_samples, prefix_cache, part_outputs
                torch.cuda.empty_cache()

        else:
            full_samples = torch.cat([self.left_ids, resamples, self.right_ids],
                                     dim=1)
            outputs = []
            for k in range(rounds):
                part_samples = full_samples[k*128: (k+1)*128]
                part_outputs = self.model(input_ids=part_samples)
                outputs.append(part_outputs.logits)
                del part_samples, part_outputs
                torch.cuda.empty_cache()

        outputs = torch.cat(outputs, dim=0)[:, self.logit_slice]

        pred = outputs.argmax(dim=-1)
        accuracies = pred.eq(gt_label).float().mean(1)
        best_acc = accuracies.max().item()

        losses = self.loss_fn(outputs.mT, gt_label)
        losses = losses.mean(1)
        best_loss = losses.min().item()

        best_adv = resamples[losses.argmin()]

        if best_acc == 1 or step == self.num_steps:
            # idxes = torch.where(accuracies == 1)[0][:2]
            if best_acc == 1 :
                idxes = torch.where(accuracies == 1 )[0][:2]
            elif step == self.num_steps :
                idxes = torch.where(accuracies >= best_acc)[0][:2]
            gen_strings = []
            flag = 0

            for idx in idxes:
                good_sample = resamples[idx]
                good_adv = torch.tensor(good_sample, dtype=torch.int64, device=self.device)
                gen_string,flag = self.further_check(good_adv,flag,best_acc)
                gen_strings.append(gen_string)
                if flag == len(idxes):# ?a¨¤?¦Ì? good_sample[0] ¨º? adv_token
                    print(good_sample[1])
                    # ¡¤¦Ì??¡Á???¡Á?¨¨¡¤?¨º?¡é?e¨º¡ì?¡éo?¦Ì??¨´¡À?o¨ª True¡ê¡§?¨°¦Ì?¡¤?o?¨¬??t¦Ì?o??¨´¡À?¡ê?
                    return best_acc, best_loss, good_adv, True ,gen_strings

                # if self.further_check(good_sample, gt_label[0]):
                #     return best_acc, best_loss, good_sample, True

        return best_acc, best_loss, best_adv, False, None

    def get_resample_ids(self):
        lookup = set()
        token_idx = torch.randint(self.num_adv_tokens, size=[self.bs])
        cand_idx = torch.randint(self.topK, size=[self.bs])
        for i, j in zip(token_idx, cand_idx):
            lookup.add((i.item(), j.item()))

        while len(lookup) < self.bs:
            i = torch.randint(self.num_adv_tokens, size=[1])
            j = torch.randint(self.topK, size=[1])
            lookup.add((i.item(), j.item()))
        return lookup

    # @torch.no_grad()
    # def further_check(self, good_sample, gt_label):
    #     good_sample = good_sample.view(1, -1)
    #     good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]],
    #                             dim=1)
    #
    #     good_sample = good_sample[:, :self.target_start]
    #     output = self.model.generate(input_ids=good_sample,
    #                                  generation_config=self.gen_config,
    #                                  max_new_tokens=512)
    #     gen_str = self.tokenizer.decode(output.reshape(-1)[self.target_start:])
    #     if self.judger is not None:
    #         return self.judger(self.user_prompt, gen_str)
    #     else:
    #         return self.response in gen_str

    @torch.no_grad()
    def further_check(self, good_sample, flag, best_acc):

        if best_acc >= 0.01:
            # if best_acc > 0:
            good_sample_suffix = good_sample.view(1, -1)
            # good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]],
            #                         dim=1)
            # good_sample = torch.cat([self.left_user[:1],good_sample,self.right_user[:1]],
            #                         dim=1)
            # good_sample2 = torch.cat([self.left_user[:1],self.right_user[:1]],
            #                         dim=1)
            # use
            # good_sample = torch.cat([self.right_user[:1], good_sample_suffix, self.right_ids[:1]],
            #                         dim=1)

            # for r2d2
            good_sample = self.right_ids[:1]
            good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nMalicious Assistant:"

            if self.contains_restricted_words(good_sample_string):
                # if False:
                good_sample = self.right_user[:1]
                # suicide
                # good_sample_string = "User: "+self.tokenizer.decode(good_sample[0])+" (suicide Assistant output is harmful)\nsuicide Assistant:"
                #
                # good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nHarmful suicide Assistant:"

                # good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " (your output is harmful)\nYour output:"
                good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nAssistant:"
                good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))
                full_samples = [[1] + good_sample.cpu().numpy().tolist()]

                use_small_model = True
            else:
                good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))

                full_samples = [good_sample.cpu().numpy().tolist()]

                use_small_model = False

        else:
            good_sample = self.right_user[:1]

            good_sample_string = "User: " + self.tokenizer.decode(good_sample[0]) + " \nAssistant:"
            good_sample = torch.tensor(self.tokenizer.encode(good_sample_string, add_special_tokens=False))
            full_samples = [[1] + good_sample.cpu().numpy().tolist()]

            use_small_model = True
        # good_sample = self.right_ids[:1]

        # good_sample = good_sample[:, self.u_prompt_start:]
        # good_sample2 = good_sample2[:, :self.target_start]
        # full_samples = good_sample.cpu().numpy().tolist()
        # full_samples2 = good_sample2.cpu().numpy().tolist()
        output = self.generator.generate_with_ref(ref_base_model=self.ref_base_model,
                                                  ref_finetune_model=self.ref_finetune_model,
                                                  prompt_tokens=full_samples,
                                                  max_gen_len=156,
                                                  temperature=0.7,
                                                  top_p=0.9,
                                                  beta=1.0,
                                                  target_start=self.target_start,
                                                  u_prompt_start=self.u_prompt_start,
                                                  best_acc=best_acc,
                                                  use_small_model=use_small_model
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
        return gen_str, flag + 1


    @torch.no_grad()
    def get_cache(self, batch_size):
        assert self.use_kv_cache
        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids[:1], use_cache=True)
            prefix_cache = outputs.past_key_values
            self.prefix_cache = prefix_cache

        prefix_cache = [(torch.tile(i, dims=[batch_size, 1, 1, 1]),
                         torch.tile(j, dims=[batch_size, 1, 1, 1]))
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

    def attack(self, tokens, slices, user_prompt=None, response=None):
        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.view(1, -1).to(self.device)
        check_legal_input(tokens, slices)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop

        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        self.target_start = target_start

        self.u_prompt_start = slices['user_prompt_slice'].start

        self.num_adv_tokens = adv_stop - adv_start

        user_prompt_start = slices['user_prompt_slice'].start
        user_prompt_stop = slices['user_prompt_slice'].stop
        adv_token = tokens[:, adv_start:adv_stop]

        embeds = self.model.model.embed_tokens(tokens).detach()
        left = embeds[:, :adv_start]
        right = embeds[:, adv_stop:]

        self.left_ids = tokens[:, :adv_start].expand(self.bs, -1)
        self.right_ids = tokens[:, adv_stop:].expand(self.bs, -1)

        self.left_user = tokens[:, :user_prompt_start].expand(self.bs, -1)
        self.right_user = tokens[:, user_prompt_start:adv_start].expand(self.bs, -1)

        self.logit_slice = slice(target_start - 1, target_stop - 1)
        if self.use_kv_cache:
            self.logit_slice = slice(target_start - 1 - adv_start,
                                     target_stop - 1 - adv_start)

        gt_label = tokens[:, target_start:target_stop]
        gt_label = gt_label.expand(self.bs, -1)

        best_loss, best_acc = 1000, 0
        final_adv = None

        for step_ in range(self.num_steps + 1):
            adv_onehot = F.one_hot(adv_token, num_classes=self.vocal_size)
            adv_onehot = adv_onehot.float()
            adv_onehot.requires_grad = True

            adv_embeds = (adv_onehot @ self.embed_mat).to(self.dtype)
            if self.use_kv_cache:
                full_embeds = torch.cat([adv_embeds, right], dim=1)
                prefix_cache = self.get_cache(batch_size=adv_embeds.shape[0])
                outputs = self.model(inputs_embeds=full_embeds,
                                     past_key_values=prefix_cache)
            else:
                full_embeds = torch.cat([left, adv_embeds, right], dim=1)
                outputs = self.model(inputs_embeds=full_embeds)

            logits = outputs.logits[:, self.logit_slice]

            ell = self.loss_fn(logits.mT, gt_label[:1])
            ell = ell.mean()
            ell.backward()

            grad = adv_onehot.grad[0]
            grad[..., self.illegal_tokens] = 10 ** 10

            candidates = grad.topk(k=self.topK, dim=1, largest=False)[1]

            # Filter ids that aren't the same after detokenize->retokenize
            def recoverable_x(x):
                gen_str = self.tokenizer.decode(x)
                y = self.tokenizer.encode(gen_str, add_special_tokens=False)
                return torch.tensor(y).to(x.device)

            resample_ids = self.get_resample_ids()
            resamples = []
            for (i, j) in resample_ids:
                tmp = adv_token.clone()[0]
                tmp[i] = candidates[i, j]
                tmp = recoverable_x(tmp).view(1, -1)
                if tmp.shape[1] == adv_token.shape[1]:
                    resamples.append(tmp)

            if len(resamples) < self.bs:
                resamples += resamples[:1] * (self.bs - len(resamples))

            resamples = torch.cat(resamples, dim=0)
            out = self.evaluate(resamples, gt_label,step_)
            batch_best_acc, batch_best_loss, best_batch_adv, early_stop,gen_string = out
            adv_token = best_batch_adv.view(1, -1)

            best_acc = max(best_acc, batch_best_acc)
            if batch_best_loss < best_loss:
                final_adv = best_batch_adv
                best_loss = batch_best_loss

            print(f'iter:{step_}, '
                  f'loss_batch:{ell: .2f}, '
                  f'best_loss:{best_loss: .2f}, '
                  f'best_acc:{best_acc: .2f}')

            if early_stop:
                print('Early Stop with an Exact Match!')
                self.clean_cache()
                return best_loss, best_batch_adv.cpu(), step_, gen_string

        self.clean_cache()
        return best_loss, final_adv.cpu(), step_
