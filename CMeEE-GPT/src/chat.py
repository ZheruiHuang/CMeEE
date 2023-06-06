import random
import openai
from src.utils.const import KEY_LIST


class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo", key="sequence", sys_setting=None, verbose=False):
        self.model = model

        self.sys_setting = "你是一个医疗领域的专家，你最擅长的任务是医疗领域命令实体识别。" if sys_setting is None else sys_setting

        self.index = 0
        self.key = key
        self.verbose = verbose

    def completion(self, prompt, answer=[], sys_setting=None):
        # ========== Configurations ==========
        self.change_api_key()
        if sys_setting is None:
            sys_setting = self.sys_setting

        # ========== Message Generation ==========
        if isinstance(prompt, str):
            prompt = [prompt]
        assert len(prompt) == len(answer) + 1, f"prompt should be one more than answer"

        messages = [{"role": "system", "content": sys_setting}]
        for i in range(len(answer)):
            messages.append({"role": "user", "content": prompt[i]})
            messages.append({"role": "assistant", "content": answer[i]})
        messages.append({"role": "user", "content": prompt[-1]})

        if self.verbose:
            print(messages)

        # ========== Chat Completion ==========
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        ans = response["choices"][0]["message"]["content"]  # type: ignore
        if self.verbose:
            print(ans)

        return ans

    def change_api_key(self):
        if self.key == "sequence":
            openai.api_key = KEY_LIST[self.index]
            self.index = (self.index + 1) % len(KEY_LIST)
        else:  # random
            openai.api_key = random.choice(KEY_LIST)


if __name__ == "__main__":
    chat = ChatGPT(key="random")
    prompt = "你好，我是一个医疗领域的专家，我最擅长的任务是医疗领域命令实体识别。"
    ans = chat.completion(prompt)
    print(ans)
