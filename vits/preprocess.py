import re, argparse, os, json
from text.cleaner import korean_cleaners

def preprocess(files: list):
    os.mkdir(f'./assets/{MODEL_NAME}')
    with open(f'./assets/{MODEL_NAME}/config.json', 'w', encoding='utf-8') as f:
        json.dump(base_json, f, indent="\t")
    for path in files:
        lines = []
        file = open(path, 'r')
        for line in file.readlines():
            mp3_path, text = line.split('|')
            text = text.replace('\n', '')
            cleaned_text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_cleaners(x.group(1))+' ', f'[KO]{text}[KO]')
            lines.append(f'{os.path.join(f"../tts_data/{MODEL_NAME}/MP3", mp3_path)}|{cleaned_text}\n')
        with open(f'./assets/{MODEL_NAME}/{path.split("/")[-1][:-4]}.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-f', '--files', nargs='+', default=['tts_data/Male/mp3_data.txt'])
    args = parser.parse_args()
    MODEL_NAME = args.model
    base_json = {
        "train": {
            "log_interval": 200,
            "epochs": 20000,
            "learning_rate": 0.0002,
            "betas": [
                0.8,
                0.99
            ],
            "eps": 1e-09,
            "batch_size": 64,
            "fp16_run": False,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "c_mel": 45,
            "c_kl": 1.0
        },
        "data": {
            "training_files": "./assets/Male/MP3_data.txt",
            "validation_files": "",
            "text_cleaners":["jk_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 44100,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "cleaned_text": True
        },
        "model": {
            "ms_istft_vits": True,
            "mb_istft_vits": False,
            "istft_vits": False,
            "subbands": 4,	
            "gen_istft_n_fft": 16,
            "gen_istft_hop_size": 4,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "upsample_rates": [4,4],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16,16],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "use_sdp": False
        }
    }
    preprocess(args.files)