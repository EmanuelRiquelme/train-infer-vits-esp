import json
import argparse

parser = argparse.ArgumentParser(description="Training script with parameters")

# Mandatory arguments
parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--batches', type=int, required=True, help='Number of batches per epoch')
parser.add_argument('--fp_16_run', type=bool, required=True, help='Whether to run in FP16 mode')
parser.add_argument('--training_files', type=str, required=True, help='Path to training files')
parser.add_argument('--validation_files', type=str, required=True, help='Path to validation files')
parser.add_argument('--output_name', type=str, required=True, help='Name of the generated Json file must include the directory')

def create_json(args):
    #creates the json file needed to train the model using the parameters recieved
    template_path = 'configs/ljs_base.json'
    with open(template_path, 'r') as file:
        template = json.load(file)
    epochs = args.epochs
    batch_size = args.batches
    fp_16 = args.fp_16_run
    training_files = args.training_files
    validation_files = args.validation_files
    output_name = args.output_name

    template['train']['epochs'] = epochs
    template['train']['batch_size'] = batch_size
    template['train']['fp16_run'] = fp_16
    template['train']['epochs'] = epochs
    template['data']['training_files'] = training_files 
    template['data']['validation_files'] = validation_files 

    with open(output_name, 'w') as file:
        json.dump(template, file, indent=2)
    print(f'File {output_name} generated sucessfully')

if __name__ == '__main__':
    args = parser.parse_args()
    create_json(args)
