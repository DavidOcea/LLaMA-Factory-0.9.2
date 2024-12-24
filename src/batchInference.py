import inspect
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.loader import _load_single_dataset
from llamafactory.data.parser import get_dataset_list
from llamafactory.extras.constants import CHOICES, SUBJECTS
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.eval.template import get_eval_template
from llamafactory.chat import ChatModel


introduction = "下面这段话中的是否包含重要的事件或声明?请从下面词语中选择一个{行业中重大政策事件/行业中影响力政策事件/行业中区域性大影响力政策事件/行业中区域性中影响力政策事件/行业中区域性小影响力政策事件/行业中不重要事件/公司经营面或信用面重大事件/公司经营面或信用面影响力事件/公司经营面或信用面大事件/公司经营面或信用面中等事件/公司经营面或信用面小事件/公司不重要事件}\n"



str2labels ={"行业中重大政策事件":10,
            "行业中影响力政策事件":8,
            "行业中区域性大影响力政策事件":6,
            "行业中区域性中影响力政策事件":4,
            "行业中区域性小影响力政策事件":2,
            "行业中不重要事件":0,
            "公司经营面或信用面重大事件":10,
            "公司经营面或信用面影响力事件":8,
            "公司经营面或信用面大事件":6,
            "公司经营面或信用面中等事件":4,
            "公司经营面或信用面小事件":2,
            "公司不重要事件":0,
}



class BatchInfer:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:

        self.model_args, self.data_args, self.eval_args, self.generation_args = get_infer_args(args)
        self.chat_model = ChatModel(args)
        self.fout = open("eval_resulets/new_evl.jsonl", "w")
        self.right = 0 
        self.wrong = 0
        self.hitlabel = 0 
        self.misslabel = 0

    def eval(self) -> None:

        # import pdb; pdb.set_trace()
         
        dataset_attrs =  get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir)
        print(self.data_args)
        self.data_args.split = 'train'

        dataset = _load_single_dataset(dataset_attrs[0], self.model_args, self.data_args, None)

        # 命中
        label0_num = 0
        label2_num = 0
        label4_num = 0
        label6_num = 0
        label8_num = 0
        label10_num = 0
        # 真值
        gt0_num = 0
        gt2_num = 0
        gt4_num = 0
        gt6_num = 0
        gt8_num = 0
        gt10_num = 0
        # 预测
        pre0_num = 0
        pre2_num = 0
        pre4_num = 0
        pre6_num = 0
        pre8_num = 0
        pre10_num = 0

        for i in trange(len(dataset), desc="Formatting batches", position=1, leave=False):
            #print(dataset[i])
            # import pdb
            # pdb.set_trace()
            messages = dataset[i]['_prompt']
            messages[0]['content'] = messages[0]['content'][:4097]
           
            response = ""
            try:
                for new_text in self.chat_model.stream_chat(messages):
                    response += new_text
            except Exception as e:
                print(e)
                #print(messages)
                #print(response)
                continue
            
            #print(response)
            #newsample ={"content": messages[0]['content'].replace(introduction,'')}
            newsample ={"content": messages[0]['content'].replace(introduction,'')[0:50]}
            newsample['label'] = -1
            #print(response)
            #print(response=="消极")
            if response.strip() in str2labels:
                newsample['label'] = str2labels[response.strip()]
                self.hitlabel +=1
            else:
                self.misslabel +=1
                

            newsample['response'] = response
            #print(newsample)

## count the right one 
            if  '_response' in dataset[i] and len(dataset[i]['_response']) ==1 :
                newsample['gt'] = str2labels[dataset[i]['_response'][0]['content'].strip()]
                # import pdb
                # pdb.set_trace()
                # if response == dataset[i]['response'][0]['content']:
                # if newsample['label'] == str2labels(dataset[i]['_response'][0]['content'].strip()):
                #     self.right +=1
                # else:
                #     self.wrong +=1
                
                newsample['gap'] = None
                if newsample['label'] != -1:
                    # gt - pre 
                    newsample['gap'] = newsample['gt'] - newsample['label']
                
                
                if newsample['label'] == 0:
                    pre0_num += 1
                if newsample['label'] == 2:
                    pre2_num += 1
                if newsample['label'] == 4:
                    pre4_num += 1
                if newsample['label'] == 6:
                    pre6_num += 1
                if newsample['label'] == 8:
                    pre8_num += 1
                if newsample['label'] == 10:
                    pre10_num += 1

                if newsample['gap'] != None:
                    if newsample['gap'] == -1 or newsample['gap'] == 0:
                        self.right +=1
                        if newsample['label'] == 0:
                            label0_num += 1
                        if newsample['label'] == 2:
                            label2_num += 1
                        if newsample['label'] == 4:
                            label4_num += 1
                        if newsample['label'] == 6:
                            label6_num += 1
                        if newsample['label'] == 8:
                            label8_num += 1
                        if newsample['label'] == 10:
                            label10_num += 1
                    else:
                        self.wrong +=1
                if newsample['gt'] == 0:
                    gt0_num += 1
                if newsample['gt'] == 1 or newsample['gt'] == 2:
                    gt2_num += 1
                if newsample['gt'] == 3 or newsample['gt'] == 4:
                    gt4_num += 1
                if newsample['gt'] == 5 or newsample['gt'] == 6:
                    gt6_num += 1
                if newsample['gt'] == 7 or newsample['gt'] == 8:
                    gt8_num += 1
                if newsample['gt'] == 9 or newsample['gt'] == 10:
                    gt10_num += 1
                

            import json 
            json.dump(newsample, self.fout, ensure_ascii=False)
            self.fout.write("\n")
            self.fout.flush()

        print("\nacc = ", self.right*1.0 / (self.right + self.wrong) )
        print("hitlabel ratio = ", self.hitlabel * 1.0 / (self.hitlabel + self.misslabel))
        print("self.right + self.wrong=",self.right + self.wrong)
        # print("-------------acc----------------")
        # print("acc_label0 = ", label0_num * 1.0 / (self.right + self.wrong) )
        # print("acc_label2 = ", label2_num * 1.0 / (self.right + self.wrong) )
        # print("acc_label4 = ", label4_num * 1.0 / (self.right + self.wrong) )
        # print("acc_label6 = ", label6_num * 1.0 / (self.right + self.wrong) )
        # print("acc_label8 = ", label8_num * 1.0 / (self.right + self.wrong) )
        # print("acc_label10 = ", label10_num * 1.0 / (self.right + self.wrong) )
        print("--------------rec---------------")
        print("rec_label0 = ", label0_num * 1.0 / gt0_num )
        print("rec_label2 = ", label2_num * 1.0 / gt2_num )
        print("rec_label4 = ", label4_num * 1.0 / gt4_num )
        print("rec_label6 = ", label6_num * 1.0 / gt6_num )
        print("rec_label8 = ", label8_num * 1.0 / gt8_num )
        print("rec_label10 = ", label10_num * 1.0 / gt10_num )
        print("--------------pre---------------")
        print("pre_label0 = ", label0_num * 1.0 / pre0_num )
        print("pre_label2 = ", label2_num * 1.0 / pre2_num )
        print("pre_label4 = ", label4_num * 1.0 / pre4_num )
        print("pre_label6 = ", label6_num * 1.0 / pre6_num )
        print("pre_label8 = ", label8_num * 1.0 / pre8_num )
        print("pre_label10 = ", label10_num * 1.0 / pre10_num )
        print("--------------data---------------")
        print("label0_num=",label0_num ," gt0_num=", gt0_num, " pre0_num=", pre0_num)
        print("label2_num=",label2_num ," gt2_num=", gt2_num, " pre0_num=", pre2_num)
        print("label4_num=",label4_num ," gt4_num=", gt4_num, " pre0_num=", pre4_num)
        print("label6_num=",label6_num ," gt6_num=", gt6_num, " pre0_num=", pre6_num)
        print("label8_num=",label8_num ," gt8_num=", gt8_num, " pre0_num=", pre8_num)
        print("label10_num=",label10_num ," gt10_num=", gt10_num, " pre0_num=", pre10_num)



        #pbar.close()
        #self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)

def main():
    BatchInfer().eval()

if __name__ == "__main__":
    main()
