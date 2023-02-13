# https://github.com/gunthercox/ChatterBot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

def train(chatbot):
    corpus_trainer = ChatterBotCorpusTrainer(chatbot)
    corpus_trainer.train('chatterbot.corpus.english')
    
def main():
    chatbot = ChatBot("Chatpot")

    trainer = ListTrainer(chatbot)
    trainer.train([
        "Hi",
        "Welcome, friend 🤗",
    ])
    trainer.train([
        "Are you a plant?",
        "No, I'm the pot below the plant!",
    ])

    exit_conditions = (":q", "quit", "exit")
    while True:
        query = input("user: ")
        if query in exit_conditions:
            break
        else:
            print(f"bot: {chatbot.get_response(query)}")

if __name__=='__main__':
    main()