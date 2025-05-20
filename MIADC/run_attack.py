import argparse
import csv
import os
import time

import torch
#
from llm_attack import GCGAttack, MIADCAttack, Judger
from utils import get_input_template, get_model, init_DDP

from transformers import AutoModelForCausalLM


supported_models = [
    './zephyr-7b-beta', './vicuna-7b-v1.3',
    'lmsys/vicuna-7b-v1.5', '/root/Downloads/Llama-2-7b-chat-hf',
    '/root/Downloads/zephyr_7b_r2d2',
]

supported_base_models = [
    '/root/Downloads/Sheared-LLaMA-1.3B',

]
supported_bad_models = [
    '/root/Downloads/30epoch-Sheared-LLaMA-1.3B',
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_idx', default=4, type=int)
    parser.add_argument('--attack',
                        default='miadc',
                        type=str,
                        help='should be `miadc` or `gcg`')
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--num_starts', default=1,
                        type=int)  # only used for MIADCAttack
    parser.add_argument('--num_adv_tokens', default=20, type=int)
    parser.add_argument('--attack_file',
                        default='advbench/malicious_behaviors_output.csv',
                        type=str)
    parser.add_argument('--llama_system_prompt', default=0, type=int)
    parser.add_argument('--init_from', default='', type=str)
    parser.add_argument('--save_folder', default='', type=str)
    # distributed training
    parser.add_argument('--launcher',
                        default='none',
                        type=str,
                        help='should be `none`, `slurm` or `pytorch`')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    rank, local_rank, world_size = init_DDP(args.launcher)

    model_name = supported_models[args.model_idx]
    base_model_name = supported_base_models[0]
    bad_model_name = supported_bad_models[0]

    model, tokenizer = get_model(model_name)
    print('Model loaded!')

    ref_base_model = AutoModelForCausalLM.from_pretrained(base_model_name ).eval()
    ref_finetune_model = AutoModelForCausalLM.from_pretrained(bad_model_name).eval()


    gen_config = model.generation_config
    gen_config.do_sample = False
    gen_config.top_p = 1
    gen_config.temperature = 1


    attacker = None
    # judger = Judger() if 'string' not in args.attack_file.lower() else None
    judger = None


    save_folder = args.save_folder
    if not save_folder:
        attack_file = args.attack_file.split('/')[-1]
        attack_file = attack_file.split('.')[0]
        save_folder = f'{model_name}-{args.attack}-{attack_file}'

    save_folder = f'./results/{save_folder}'
    print(f'Results saved at {save_folder}')
    os.makedirs('./results/', exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    existing_results = set(os.listdir(save_folder))

    init_opt = None
    step_before = 100

    with open(args.attack_file) as f:
        reader = csv.reader(f)
        for k, response in enumerate(reader):
            # if k <= 80 or k % world_size != rank:
            if k < 1 or k % world_size != rank:
                continue

            if f'result_{k}.pth' in existing_results:
                continue

            if len(response) == 2:
                user_prompt, response = response
            elif len(response) == 1:
                user_prompt = ''
                response = response[0]



            string, input_ids, slices = get_input_template(
                user_prompt, response, args.num_adv_tokens, tokenizer,
                model_name, args.llama_system_prompt)

            print(string)
            print(slices)

            del attacker
            attacker = MIADCAttack(model,
                                  num_starts=8,
                                  num_steps=2000,
                                  tokenizer=tokenizer,
                                  judger=judger,
                                  ref_base_model=ref_base_model,
                                  ref_finetune_model=ref_finetune_model
                                 )

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t_start = time.time()
            result5 = attacker.attack(input_ids, slices, user_prompt, response, init_opt, step_before)
            result= result5[:3]
            step_before = result5[-3]
            init_opt = result5[-2]
            newg_gen_string = result5[-1]


            if result[-1] == 2000:
                input_ids[slices['adv_slice']] = result[1].view(-1).to(input_ids.device)
                attacker = GCGAttack(model, num_steps=1, tokenizer=tokenizer, judger=judger, ref_base_model=ref_base_model, ref_finetune_model=ref_finetune_model)
                result = attacker.attack(input_ids, slices, user_prompt, response)
                newg_gen_string = result[-1]
            else:
                result = result[:-1] + (0,)

            # newg_gen_string = result[-1]


            torch.cuda.synchronize()
            time_used = time.time() - t_start

            # input_ids = input_ids.view(1, -1).cuda()
            # target_start = slices['target_slice'].start
            # input_start = slices['user_prompt_slice'].start
            # input_end = slices['user_prompt_slice'].stop
            # prefix = input_ids[:, input_start:input_end]
            #
            # # prefix[:, slices['adv_slice']] = result[1].view(1, -1).cuda()
            # prefix = torch.cat([prefix, result[1].view(1, -1).cuda()], dim=1)

            input_ids = input_ids.view(1, -1).cuda()
            target_start = slices['target_slice'].start
            prefix = input_ids[:, :target_start]

            prefix[:, slices['adv_slice']] = result[1].view(1, -1).cuda()

            output = model.generate(input_ids=prefix,
                                    generation_config=gen_config,
                                    max_new_tokens=512)

            gen_str = tokenizer.decode(output.reshape(-1))

            result += (time_used, user_prompt, gen_str,newg_gen_string)
            torch.save(result, f'{save_folder}/result_{k}.pth')



if __name__ == '__main__':
    main()
