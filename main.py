import nsml
from nsml import HAS_DATASET, DATASET_PATH
from dialogue_summarization.models import SummarizationModel
import argparse
import os
from glob import glob
import json
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

import json
from sklearn.model_selection import train_test_split
from dialogue_summarization.datasets import SummarizationDataset, text_infil
from transformers import (
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    BertTokenizerFast, 
    BertModel,
    BertTokenizerFast, 
    EncoderDecoderModel,
    EncoderDecoderConfig,
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    PegasusConfig,
    MBartConfig,
    MBartForConditionalGeneration,
)
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings(action='ignore')


def train_data_loader(root_path):
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

@staticmethod
def make_dataset_list(path_list):
    json_data_list = []

    for path in path_list:
        with open(path) as f:
            json_data_list.append(json.load(f))

    return json_data_list



# def nsml_save_asap(filepath, **kwargs):
#     print(os.path.join(filepath))
#     model.save_pretrained(os.path.join(filepath, 'model.ckpt'))

#     # state = {
#     # 'state_dict': model.state_dict()
#     # }

#     # torch.save(state, os.path.join(filepath, 'model.ckpt'))
#     print("저장 완료!")

def load(dirname, *args):
    
    # global model
    # model = MBartForConditionalGeneration.from_pretrained(os.path.join(dirname, 'model.ckpt'))

    checkpoint = os.path.join(dirname, "model.ckpt")
    global model
    #model_config = MBartConfig.from_json_file("configs/mbart_config.json")
    #model = MBartForConditionalGeneration(model_config)
    model_config = BartConfig.from_json_file("configs/kobart_config.json")
    model = BartForConditionalGeneration(model_config)

    old = torch.load(checkpoint)['state_dict']
    keys = old.keys()
    new={}
    for i in keys:
        new[i[6:]] = old[i]
    
    model.load_state_dict(new)

    print("로딩 완료!")

# def load(dirname, *args):
#     # ## 2) trainer에서 저장한 ckpt 로드하고 싶을 때
#     checkpoint = os.path.join(dirname, "model.ckpt")
#     # print(os.listdir(checkpoint))

#     # child_p = os.listdir(checkpoint)

#     # r_file = os.listdir(os.path.join(dirname, "model.ckpt",child_p))
#     r_checkpoint = os.path.join(dirname, "model.ckpt",'pytorch_model.bin')
#     print(r_checkpoint)
#     global model
#     model_config = MBartConfig.from_json_file("configs/mbart_config.json")
#     model = MBartForConditionalGeneration(model_config)
#     # model.from_pretrained(checkpoint)
 
#     old = torch.load(r_checkpoint)
#     print(old.keys())
    
#     # keys = old.keys()
#     # new={}
#     # for i in keys:
#     #     new[i[13:]] = old[i]

#     model.load_state_dict(old)
#     print("로딩 완료!")

def infer(test_path, **kwparser):
    device = torch.device("cuda:0")

    test_json_path = os.path.join(test_path, 'test_data', '*')

    print(f'test_json_path :\n{test_json_path}')
    test_path_list = glob(test_json_path)
    test_path_list.sort()
    print(f'test_path_list :\n{test_path_list}')
    print('종류 : ', len(test_path_list))#5
    # eda

    config = SummarizationModel.load_config("../configs/", "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_all = []
    for filename in test_path_list:
        print(filename, len(json.load(open(filename, "r"))["data"]))
        data_all += json.load(open(filename, "r"))["data"]
    print("전체데이터 합치기 완료")

    test_dataset = SummarizationDataset(
        #filename=self.config.valid_data_path,
        datafile=data_all,
        separator=config.separator,
        meta_sep=config.meta_sep,
        tokenizer=tokenizer,
        max_length=config.max_length,
        strategy=config.strategy,
        is_test=1
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=os.cpu_count()
    )

    #del test_dataset
    tqdm_dataset = tqdm(enumerate(test_dataloader))

    test_ids = []
    summary = []

    model.to(device)

    for batch, batch_item in tqdm_dataset:##
        print(batch)
        test_id = batch_item['dialogueID']

        output = model.generate(batch_item['input_ids'].to(device), eos_token_id=6, max_length=50, top_k=200, num_beams=8, 
            num_return_sequences=1,
                    no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=0.7)  ## todevice


        output = output.cpu()

        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        ## for test
        if batch < 10:
            #print(output)
            print(list(zip(test_ids, summary)))
        

        test_ids.extend(test_id)
        summary.extend(output)

    # #cleaning
    # clean_summary = []
    # for i in summary:
    #     clean_summary.append(i.split('.')[0] + '.')

    # DONOTCHANGE: They are reserved for nsml
    # 리턴 결과는 [(id, summary)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
    # return list(zip(pred.flatten(), clipped.flatten()))
    #return list(zip(test_ids, clean_summary))  ##
    return list(zip(test_ids, summary))
        ## ex)[(' efe21026-0715-5ca4-99fe-46d0ecfba147', '철수는 밥을 먹었다.'), ...]

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.

# nsml.bind(load=load, infer=infer, save=nsml_save_asap)
nsml.bind(load=load, infer=infer)



def train_infer(test_path):
    device = torch.device("cuda:0")

    #test_json_path = os.path.join(test_path, 'test_data', '*')
    test_json_path = os.path.join(test_path, 'train', 'train_data', '*')

    print(f'test_json_path :\n{test_json_path}')
    test_path_list = glob(test_json_path)
    test_path_list.sort()
    print(f'test_path_list :\n{test_path_list}')
    print('종류 : ', len(test_path_list))#5
    # eda

    config = SummarizationModel.load_config("../configs/", "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_all = []
    for filename in test_path_list:
        print(filename, len(json.load(open(filename, "r"))["data"]))
        data_all += json.load(open(filename, "r"))["data"]
    print("전체데이터 합치기 완료")

    test_dataset = SummarizationDataset(
        #filename=self.config.valid_data_path,
        datafile=data_all,
        separator=config.separator,
        meta_sep=config.meta_sep,
        tokenizer=tokenizer,
        max_length=config.max_length,
        strategy=config.strategy,
        is_test=1
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    #del test_dataset
    tqdm_dataset = tqdm(enumerate(test_dataloader))

    test_ids = []
    summary = []

    model.to(device)

    for batch, batch_item in tqdm_dataset:##
        print(batch)
        test_id = batch_item['dialogueID']

        output = model.generate(batch_item['input_ids'].to(device), eos_token_id=6, max_length=50, top_k=200, num_beams=8, 
            num_return_sequences=1,
                    no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=0.7)  ## todevice


        output = output.cpu()

        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        ## for test
        if batch < 10:
            #print(output)
            print(list(zip(test_ids, summary)))
        

        test_ids.extend(test_id)
        summary.extend(output)

    return list(zip(test_ids, summary))



def train_infer_infil(test_path):
    device = torch.device("cuda:0")

    #test_json_path = os.path.join(test_path, 'test_data', '*')
    test_json_path = os.path.join(test_path, 'train', 'train_data', '*')

    print(f'test_json_path :\n{test_json_path}')
    test_path_list = glob(test_json_path)
    test_path_list.sort()
    print(f'test_path_list :\n{test_path_list}')
    print('종류 : ', len(test_path_list))#5
    # eda

    config = SummarizationModel.load_config("../configs/", "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_all = []
    for filename in [test_path_list[0]]:   ## 1개 아닐 경우 수정 필요**
        print(filename, len(json.load(open(filename, "r"))["data"]))
        data_all += json.load(open(filename, "r"))["data"]
    print("전체데이터 합치기 완료")

    test_dataset = SummarizationDataset(
        #filename=self.config.valid_data_path,
        datafile=data_all,
        separator=config.separator,
        meta_sep=config.meta_sep,
        tokenizer=tokenizer,
        max_length=config.max_length,
        strategy=config.strategy
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
        #collate_fn=SummarizationModel.collate_fn,
    )

    #del test_dataset
    tqdm_dataset = tqdm(enumerate(test_dataloader))

    test_ids = []
    summary = []

    model.to(device)

    for batch, batch_item in tqdm_dataset:##
        print(batch)
        test_id = batch_item['dialogueID']

        output = model.generate(batch_item['input_ids'].to(device), eos_token_id=6, max_length=50, top_k=200, num_beams=8, 
            num_return_sequences=1,
                    no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=0.7)  ## todevice


        output = output.cpu()

        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        ## for test
        if batch < 10:
            #print(output)
            print(list(zip(test_ids, summary)))
        

        test_ids.extend(test_id)
        summary.extend(output)

    return list(zip(test_ids, summary))


def train_data_loader(root_path):
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

@staticmethod
def make_dataset_list(path_list):
    json_data_list = []

    for path in path_list:
        with open(path) as f:
            json_data_list.append(json.load(f))

    return json_data_list


def check_sample(model):
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dialogue = "북구청 때문에 너희 집 앞이 너무 무서워북구청은 의심환자만 은밀하게가코로나 동선겹칠까봐그닥 안무서워 어딜가나 똑같어동선 안겹쳐의심환자들은 우리가지나가는길로안가뒤에주차장들어가는데 따로 구석진곳이 있어 "
    model.eval().to(device)
    input_ids = tokenizer(dialogue, return_tensors="pt",max_length=300,padding="max_length").input_ids[0]
    input_ids = tokenizer.encode(dialogue)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)

    output = model.generate(input_ids.to(device), 
        eos_token_id=6, max_length=50, top_k=200, num_beams=5, 
            num_return_sequences=1,
                    no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=0.7)  ## todevice

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output)
    return output

def check_sample_infil(model):
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dialogue = "북구청 때문에 너희 집 앞이 너무 무서워북구청은 의심환자만 은밀하게가코로나 동선겹칠까봐그닥 안무서워 어딜가나 똑같어동선 안겹쳐의심환자들은 우리가지나가는길로안가뒤에주차장들어가는데 따로 구석진곳이 있어 "
    model.eval().to(device)
    #input_ids = tokenizer(dialogue, return_tensors="pt",max_length=300,padding="max_length").input_ids[0]
    input_ids = tokenizer(dialogue, return_tensors="pt",max_length=300,padding="max_length").input_ids
    #input_ids = tokenizer.encode(dialogue)
    #input_ids = torch.tensor(input_ids)
    #input_ids = input_ids.unsqueeze(0)
    input_ids, _ = text_infil(tokenizer, input_ids, special_tokens_mask=None)
    print(input_ids)

    output = model.generate(input_ids[0][0].to(device), 
        eos_token_id=6, max_length=50, top_k=200, num_beams=5, 
            num_return_sequences=1,
                    no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=0.7)  ## todevice

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()


    config = SummarizationModel.load_config("../configs/", "config.yaml")

    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        #global model
        #model = BartForConditionalGeneration(BartConfig.from_pretrained(config.model_name))

        #nsml.load(checkpoint=10, session='nia2016/final_dialogue/313')
        # nsml.load(checkpoint=22, session='nia2016/final_dialogue/323')
        #nsml.load(checkpoint="best", session='nia2016/final_dialogue/357')
        #nsml.load(checkpoint="best", session='nia2016/final_dialogue/404')
        #nsml.load(checkpoint=207, session='nia2016/final_dialogue/412')
        nsml.load(checkpoint=69, session='nia2016/final_dialogue/459')
        #train_infer_infil(DATASET_PATH)
        print(check_sample_infil(model))


        exit()
        ## 토크나이저 작동테스트
        #tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        #dialogue = "북구청 때문에 너희 집 앞이 너무 무서워[SEP]북구청은 의심환자만 은밀하게가[SEP]코로나 동선겹칠까봐그닥 안무서워 어딜가나 똑같어동선 안겹쳐의심환자들은 우리가지나가는길로안가뒤에주차장들어가는데 따로 구석진곳이 있어 "
        #input_ids = tokenizer(dialogue, return_tensors="pt",max_length=300,padding="max_length").input_ids[0]
        #print(input_ids)
        #output = tokenizer.decode(input_ids, skip_special_tokens=True)
        #print(output)

        #print("load test: ", check_sample(model))
        #results = train_infer(DATASET_PATH)

        #nsml.save("best")
        #exit()

        #nsml.load(checkpoint=21, session='nia2016/dialogue/936')
        # nsml.save('checkpoint')

        #nsml.load(checkpoint=11, session='nia2016/dialogue/1143')
        #nsml.load(checkpoint=6, session='nia2016/dialogue/1143')
        #nsml.load(checkpoint=11, session='nia2016/final_dialogue/1')
        #nsml.save("best")
        

        ###################################################################
        ################ choose model #####################################
        my_model = "bart"
        my_model = "minibart"
    
        if my_model == "bart":
            model_config = BartConfig.from_json_file("configs/kobart_config.json")
            model = BartForConditionalGeneration(model_config)  

        elif my_model == "pegasus":
            model_config = PegasusConfig.from_json_file("configs/pegasus_config.json")
            model = PegasusForConditionalGeneration(model_config)  
        
        elif my_model == "mbart":
            model_config = MBartConfig.from_json_file("configs/mbart_config.json")
            model = MBartForConditionalGeneration(model_config)
        ###################################################################

        #exit()
        print("start !! ")
        train_path_list = train_data_loader(DATASET_PATH)
        train_path_list.sort()
        #print(train_path_list)

        data_all = []
        for filename in [train_path_list[0]]:
            data_all += json.load(open(filename, "r"))["data"]
        print("전체데이터 합치기 완료")

        
        train_data = []
        valid_data = []

        # # train_path_list = [train_path_list[0]]   # sampling
        for filename in train_path_list:
            # filename.split('/')[-1].split('.json')[0]
            temp_data = json.load(open(filename, "r"))["data"]
            t, v = train_test_split(temp_data, test_size=0.1, shuffle=True, random_state=42)
            train_data += t
            valid_data += v
        
            
        # config = SummarizationModel.load_config("../configs/", "config.yaml")
        # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # model = SummarizationModel("../configs/", "config.yaml", train_data, valid_data, pre_model=model)  ## path -> data ([{'body': 0000}, …] 로 수정

        # train_dataset = SummarizationDataset(
        #     #filename=self.config.valid_data_path,
        #     datafile=train_data,
        #     separator=config.separator,
        #     meta_sep=config.meta_sep,
        #     tokenizer=tokenizer,
        #     max_length=config.max_length,
        #     strategy=config.strategy,
        #     is_test=1
        # )

        # train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=config.batch_size,
        #     drop_last=True,
        #     pin_memory=True,
        #     num_workers=os.cpu_count(),
        #     collate_fn=model.collate_fn
        # )

            
        # tqdm_dataset = tqdm(enumerate(train_dataloader))

        # for batch, batch_item in tqdm_dataset:##
        #     print(batch)
        #     #output = tokenizer.batch_decode(batch_item['input_ids'], skip_special_tokens=True)
            
        #     ## for test
        #     if batch < 10:
        #         #print(output)
        #         print(batch_item)
        #         #print(output)
            

        #data_all = json.load(open(train_path_list[0], "r"))["data"]
        #train_data, valid_data = train_test_split(data_all, test_size=0.1, shuffle=True, random_state=42)


        #전체데이터 활용시
        #train_data = data_all
        print("train valid split 완료")

        model = SummarizationModel("../configs/", "config.yaml", train_data, valid_data, pre_model=model)  ## path -> data ([{'body': 0000}, …] 로 수정

        model.fit()