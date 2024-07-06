# Encoder + Decoder的做法, 主要想法是吧encoder的内容以encoder_hidden_states的方式给decoder
import os, torch, csv
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # if using Apple chips, i.e. device = mps

from datasets import MFRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.optim import Adam
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, AutoImageProcessor, AutoTokenizer, AutoConfig
import numpy as np
import random

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def freeze(model, do_train=False):
    for k, v in model.named_parameters():
        v.requires_grad = do_train

def count_trainable(model):
    model.train()
    cnt = 0
    for k, v in model.named_parameters():
        if v.requires_grad:
            print(k, v.shape)
            cnt += v.numel()
    print(f"Total Trainable Param {cnt / 10 ** 6:.2f} M")  # 247.4M

def preprocess_img(img):
    # 训练和测试都经过同样的preprocess_img
    return img.crop((140, 430, 1140, 530))

def collate_fn(batch):
    x1 = img_processor([preprocess_img(item[0]) for item in batch],
                       return_tensors="pt").pixel_values  # ,padding=True
    x = tokenizer([item[1] for item in batch], return_tensors="pt", padding=True).input_ids
    # 默认会在x末尾增加eos_token_id，开头增加bos_token_id
    # x = torch.cat([x,torch.tensor([tokenizer.bos_token_id]).repeat(x.size(0),1)],dim=-1) # (N,L-1) (N,1) -> (N,L)
    return x1, x

def evaluate(dataset,output_path):
    # 如果两个字符串完全相同 -> 正确!
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["pred_res", "y_pred", "y", "correct"])

        correct = 0
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataset):
                x = img_processor(preprocess_img(x), return_tensors="pt").pixel_values.to(device)
                # generate默认第一个输入的token是tokenizer.bos_token (根据model.generation_config来的)
                res = model.generate(pixel_values=x, max_new_tokens=100, do_sample=False)
                res = tokenizer.decode(res[0])
                y_pred = res.split("</s>")[0].replace(" ", "")
                is_correct = (y.replace(" ", "") == y_pred)
                correct += is_correct

                # 将结果写入CSV文件
                writer.writerow([res, y_pred, y, is_correct])

    return correct / len(dataset)

if __name__ == "__main__":
    lr = 5e-5
    bs = 1
    num_epochs = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device",device)
    set_seed(42)
    CUR_DIR = os.path.dirname(__file__)
    output_path = f"{CUR_DIR}/models"
    os.makedirs(output_path,exist_ok=True)
    # 训练记录
    history = []  # 记录 ['Epoch', 'Step', 'Train Loss'] # 'Train Accuracy', 'Dev Loss', 'Dev Accuracy'
    csv_file = f'{output_path}/training_log.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Step', 'Train Loss', 'Accuracy'])

    # 导入数据
    # 以latex公式的数据为例
    train = MFRDataset(csv_path=f"{CUR_DIR}/data/MathFormulaRecognition/split_latex/train.csv",
                       pic_folder=f"{CUR_DIR}/data/MathFormulaRecognition/train_latex")
    dev = MFRDataset(csv_path=f"{CUR_DIR}/data/MathFormulaRecognition/split_latex/dev.csv",
                       pic_folder=f"{CUR_DIR}/data/MathFormulaRecognition/train_latex")
    # train = dev
    # test = MFRDataset(csv_path=f"{CUR_DIR}/data/MathFormulaRecognition/split_latex/test.csv",
    #                  pic_folder=f"{CUR_DIR}/data/MathFormulaRecognition/train_latex")

    # 导入模型
    img_encoder = 'microsoft/swin-base-patch4-window7-224-in22k'  # 'openai/clip-vit-base-patch32'
    text_decoder = 'facebook/bart-base'  # 'Qwen/Qwen1.5-0.5B', 'facebook/mbart-large-50-many-to-many-mmt'

    img_processor = AutoImageProcessor.from_pretrained(img_encoder)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder)

    img_encoder_config = AutoConfig.from_pretrained(img_encoder)
    text_decoder_config = AutoConfig.from_pretrained(text_decoder)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(img_encoder_config, text_decoder_config)

    model = VisionEncoderDecoderModel(config=config)

    # 如果encoder和decoder的hidden_dim相等不一定增加enc_to_dec_proj，可以手动写
    # model.enc_to_dec_proj = nn.ModuleList([
    #     nn.Linear(model.config.encoder.hidden_size, 2048),
    #     nn.GELU(),
    #     nn.Linear(2048, model.config.decoder.hidden_size)])

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.decoder.config.decoder_start_token_id = tokenizer.bos_token_id # bart默认的是</s>..
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id

    # freeze(model)
    # freeze(model.enc_to_dec_proj,do_train=True)
    # freeze(model.decoder.lm_head,do_train=True)
    count_trainable(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    # 正式训练模型
    log_interval = 100  # 每隔多少步记录一次, 2
    step = 0
    running_loss = 0  # 每log_interval个step归零
    total_sample = 0
    correct = 0

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    # dev_loader = DataLoader(dev, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        model.train()
        for _, (img, text) in tqdm(enumerate(train_loader)):
            img = img.to(device)
            text = text[:,1:].to(device) # 因为model在计算损失的时候会自己增加一个tokenizer.bos_token_id
            res = model(pixel_values=img, labels=text)
            # 上述代码等价于
            # set_seed(42)  # 因为内部会有随机的东西，为了比较结果一致性需要增加这个
            # out = model.encoder(pixel_values=img).last_hidden_state
            # out1 = model.enc_to_dec_proj(out)
            # new_text = torch.cat([torch.tensor([tokenizer.bos_token_id]).repeat(text.size(0), 1), text[:, :-1]], dim=-1)
            # out2 = model.decoder(input_ids=new_text, labels=text, encoder_hidden_states=out1)

            loss = res.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_sample += img.size(0)
            correct += (res.logits.argmax(dim=-1)==text).sum().item()
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                accuracy = correct / total_sample

                # 将训练结果写入CSV文件
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, step, avg_loss, accuracy])

                running_loss = 0.0  # 重置损失
                total_sample = 0
                correct = 0

                # 在验证集上验证
                dev_accuracy = evaluate(dev, f'{output_path}/dev-step{step}.csv')

                print(f'Epoch [{epoch + 1}/{num_epochs}], \
                      Step [{step}], Loss: {avg_loss:.3f},\
                      Train Acc: {accuracy:.3f},\
                      Dev Acc: {dev_accuracy:.3f}')
                torch.save(model.state_dict(), f"{CUR_DIR}/model.pt")
