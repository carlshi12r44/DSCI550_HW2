import gpt_2_simple as gpt2
import os
import requests

def generate_text_gpt2(input_text_file_path, save_parent_path, save_folder,model_name):
    """
    generate text by gpt 2

    """
    sess = gpt2.start_tf_sess()
    # run each type for 10 times
    gpt2.finetune(sess,input_text_file_path,model_name=model_name,steps=15)   # steps is max number of training steps

    gpt2.load_gpt2(sess)

    for i in range(200):
        single_text = gpt2.generate(sess, return_as_list=True)[0]
        fw = open(save_parent_path + save_folder + f"{i}.txt", "w+")
        fw.write(single_text)
        fw.close()

if __name__ == "__main__":
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
    
    file_name = "shakespeare.txt"
    if not os.path.isfile(file_name):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data = requests.get(url)
        
        with open(file_name, 'w') as f:
            f.write(data.text)

    reconnaissance_path = os.getcwd() + "/data/part4/Reconnaissance/total.txt"
    malware_path = os.getcwd() + "/data/part4/Malware/total.txt"
    socialEngineering_path = os.getcwd() + "/data/part4/SocialEngineering/total.txt"
    credentialPhishing_path = os.getcwd() + "/data/part4/CredentialPhishing/total.txt"

    # sess = gpt2.start_tf_sess()

    # #gpt2.finetune(sess,file_name,model_name=model_name,steps=10)   # steps is max number of training steps
    # gpt2.load_gpt2(sess)
    # single_text = gpt2.generate(sess,return_as_list=True)[0]
    # print(single_text)
    generate_text_gpt2(reconnaissance_path, os.getcwd() + '/data/part4/Reconnaissance/','GPT2_res/',model_name)