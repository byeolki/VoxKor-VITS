import re, argparse, os, shutil
from text.cleaner import korean_cleaners

def preprocess(files: list):
    os.mkdir(f'./outputs/{MODEL_NAME}')
    shutil.copyfile('./src/assets/config.json', f'./outputs/{MODEL_NAME}/config.json')
    for path in files:
        lines = []
        file = open(path, 'r')
        for line in file.readlines():
            mp3_path, text = line.split('|')
            text = text.replace('\n', '')
            cleaned_text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_cleaners(x.group(1))+' ', f'[KO]{text}[KO]')
            lines.append(f'{os.path.join(f"../data/{MODEL_NAME}/MP3", mp3_path)}|{cleaned_text}\n')
        with open(f'./outputs/{MODEL_NAME}/{path.split("/")[-1][:-4]}.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-f', '--files', nargs='+', default=['data/Male/mp3_data.txt'])
    args = parser.parse_args()
    MODEL_NAME = args.model
    preprocess(args.files)