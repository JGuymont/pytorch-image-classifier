import subprocess

MAIN_FILE = './example/fashion-mnist/main.py'
INPUT_SIZE = "28"
INPUT_CHANNEL = "1" # greyscale
NUM_CLASSES = "10"

MODEL_NAME = "CNN-V2"

COMMAND = [
    "python", MAIN_FILE,
    "--input_size", INPUT_SIZE,
    "--num_classes", NUM_CLASSES,
    '--input_channel', INPUT_CHANNEL,

    "--model_name", MODEL_NAME,
    '--channel_sizes', "64 32",               
    '--kernel_sizes', "2 2",            
    '--dropout', "0.3 0.3",               
    '--max_pool', "2 2",                 
    '--batch_norm', "0 1",

    '--num_dense_layer', "2",
    '--hidden_layer_sizes', "256",        
    '--dense_dropout', "0.5",
    '--epochs', "10",                                            
    '--lr', "0.01",
    '--momentum', "0.5",
]

if __name__ == '__main__':


    subprocess.call(COMMAND)

