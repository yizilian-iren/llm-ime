"""English words the engine offers as typed when the keys are not pinyin (5B.9: github, cpu, tracker) -> ime/en_words.txt.

  python scripts/build_en_words.py path/to/scowl-2020.12.07/final

SCOWL (http://wordlist.aspell.net, Kevin Atkinson; its Copyright file is copied into the output, as its permission
asks): english-words and american-words up to size 50, lowercase letters only, at least 3 letters and with a vowel:
by / my / hmm / why are pinyin initials far more often (不要, 没有). Then TECH, technical words no dictionary has.
"""

import glob
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ime", "en_words.txt")
LEVELS = (10, 20, 35, 40, 50)
TECH = """
github gitlab git repo readme commit rebase diff todo debug config localhost
cpu gpu ram ssd usb hdmi wifi vpn dns ssh http https url html css json yaml toml csv xml sql pdf png jpg jpeg gif svg mp mov
cmd sudo bash zsh tmux vim nvim emacs grep curl wget brew npm npx pnpm yarn pip conda venv uv
python docker kubernetes nginx redis mysql postgres sqlite mongodb linux ubuntu debian macos ios ipad iphone xcode swift
rust golang java javascript typescript nodejs react vue svelte nextjs vite webpack eslint pytest jupyter numpy pandas
pytorch torch cuda nvidia dgx mlx llm llms gpt chatgpt openai anthropic claude gemini qwen deepseek llama ollama huggingface
lora sft dpo rlhf api apis sdk cli gui ide vscode cursor copilot slack notion figma obsidian chrome safari firefox wechat
ppt pptx docx xlsx app apps ok
"""


def main():
    final = sys.argv[1]
    words = set()
    for kind in ("english-words", "american-words"):
        for lv in LEVELS:
            for path in glob.glob(os.path.join(final, f"{kind}.{lv}")):
                for line in open(path, encoding="latin-1"):
                    w = line.strip()
                    if re.fullmatch(r"[a-z]{3,}", w) and re.search("[aeiou]", w):
                        words.add(w)
    words |= set(TECH.split())
    notice = open(os.path.join(os.path.dirname(final.rstrip("/")), "Copyright"), encoding="latin-1").read().strip()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# English words for 5B.9, scripts/build_en_words.py: SCOWL english/american-words up to size 50 "
                "plus technical words.\n")
        f.write("".join(f"# {line}\n".replace("# \n", "#\n") for line in notice.splitlines()))
        f.write("\n".join(sorted(words)) + "\n")
    print(f"{len(words)} words -> {OUT}")


if __name__ == "__main__":
    main()
