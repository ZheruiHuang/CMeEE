import os
import json
import time

from src.utils.etqdm import etqdm
from src.utils.utils import back_to_data
from src.utils.logger import TxtLogger
from src.utils.settings import get_json_data, set_proxy
from src.prompt.get_prompt import get_prompt
from src.cluster.get_cluster import get_cluster
from src.chat import ChatGPT
from src.args import GPTParser


def main(args):
    setting = "你是一个医疗领域的专家。"
    setting += "请你处理一个医疗领域命名实体识别的任务，可以抽取的实体类别包括疾病、临床表现、医疗程序、医疗设备、药物、医学检验项目、身体、科室、微生物一共九类。"
    setting += "对于输入的文本，请你按顺序抽取出其中的命名实体，输出所属类别。\n"

    # =========== Configurations ===========
    if args.VPN:
        set_proxy(args.port)
    all_data = get_json_data(args.data)
    prompt_gen = get_prompt(args.prompt)
    shot_chooser = get_cluster(mode=args.c)
    print(f"===> Mode: {args.c}")
    print(f"===> Number of Shots: {args.num_shots}")
    print(f"===> Prompt: {args.prompt}")

    # =========== Output File ===========
    if args.Debug:
        name = "debug"
    else:
        name = f"{args.c}_{args.num_shots}_{args.prompt}_{args.data}"
        idx = 1
        while os.path.exists(os.path.join("results", f"{name}.json")):
            name = f"{args.c}_{args.num_shots}_{args.prompt}_{args.data}_{idx}"
            idx += 1
    output_file = os.path.join("results", f"{name}.json")
    logger = TxtLogger(name, "logs")
    print(f"===> Output File: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as result_file:
        result_file.write("[\n")

    # =========== ChatGPT ===========
    chat = ChatGPT()
    length = len(all_data)
    for i in etqdm(range(length)):
        data = all_data[i]
        text = data["text"]

        # =========== Shot Choosing ===========
        shot = shot_chooser.get_shots(text, args.num_shots)
        example_question = prompt_gen(shot, text)
        prompt = "\n".join([setting, example_question])
        logger.info(prompt)

        # =========== ChatGPT Completion ===========
        while True:
            try:
                response = chat.completion(prompt)
                pred = back_to_data(response, text)
                break
            except Exception as e:
                print(e)
                time.sleep(0.2)
                continue
        logger.info(response)
        data_dict = {"text": text, "entities": pred}

        # =========== Save Results ===========
        with open(output_file, "a", encoding="utf-8") as result_file:
            result_file.write(json.dumps(data_dict, indent=4, sort_keys=False, ensure_ascii=False))
            if i != length - 1:
                result_file.write(",\n")
            else:
                result_file.write("\n")

        if args.Debug:
            break

        time.sleep(0.2)

    with open(output_file, "a", encoding="utf-8") as result_file:
        result_file.write("]")


if __name__ == "__main__":
    args = GPTParser()
    main(args)
