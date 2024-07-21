import argparse
from colorama import init, Fore, Style
from pyfiglet import Figlet

init(autoreset=True)

def print_colored_text():
    fig = Figlet(font='slant')  
    tool_name_art = fig.renderText("AIducator")

    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
    colored_tool_name = ''.join(
        f"{colors[i % len(colors)]}{char}"
        for i, char in enumerate(tool_name_art)
    )
    
    print(f"{colored_tool_name}")

def prompt_user():
    prompt_text = "What would you like to learn today?"
    fig = Figlet(font='digital')  
    prompt_art = fig.renderText(prompt_text)
    prompt_color = Fore.CYAN
    colored_prompt = f"{prompt_color}{prompt_art}"
    print(colored_prompt)
    user_input = input(Fore.BLUE + "Your answer: ")
    return user_input

argparse = argparse.ArgumentParser(description="AIducator", usage="Your one stop shop for modern learning :)")
argparse.add_argument("-c", "--chatbot", help="Starts a fully customised chatbot for solving your Doubts suiting your environment", required=False)
args = argparse.parse_args()
chatbot = args.chatbot

print_colored_text()

user_response = prompt_user()
print(f"{Fore.GREEN}You entered: {user_response}")

