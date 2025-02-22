from openai import OpenAI

client = OpenAI()


def initialize_messages() -> list:
    """Initialize the chat messages with system and user messages."""
    # TODO; since this prompt is not sufficient in steering the bot to use only the custom knowledge, experiment with it.
    return [
        {"role": "developer", 
         "content": "Šta podrazumevaju osnovne radionice za digitalnu pismenost?"},
        {"role": "assistant", 
         "content": "Osnovne radionice za digitalnu pismenost obuhvataju savladavanje korišćenja pametnih telefona i tableta, otvaranje e-mail naloga, kao i osnove rada na računaru, uključujući pisanje dokumenata i pretragu interneta."},
        {"role": "developer", 
         "content": "Šta podrazumevaju napredne obuke za e-usluge?"},
        {"role": "assistant", 
         "content": "Napredne obuke za e-usluge uključuju korišćenje portala eUprave (plaćanje računa, zakazivanje termina, preuzimanje dokumenata), onlajn bankarstvo i upotrebu aplikacije eAgrar."},
        {"role": "developer", 
         "content": "Šta podrazumevaju praktični treninzi?"},
        {"role": "assistant", 
         "content": "Praktični treninzi obuhvataju obuku za bezbednu onlajn kupovinu i korišćenje dostavnih servisa."},
        {"role": "developer", 
         "content": "Šta je DIGIMOST?"},
        {"role": "assistant", 
         "content": "DIGIMOST je učenička kompanija koja pruža prilagođene obuke o korišćenju digitalnih alata starijim sugrađanima i svima kojima je potrebno osamostaljivanje u digitalnom okruženju."},
        {"role": "developer", 
         "content": "Gde se nalazi DIGIMOST?"},
        {"role": "assistant", 
         "content": "Učenička kompanija DIGIMOST nalazi se u Školi Petro Kuzmjak u Ruskom Krsturu."},
        {"role": "developer", 
         "content": "Gde se održavaju obuke?"},
        {"role": "assistant", 
         "content": "Obuke se organizuju u Školi &quot;Petro Kuzmjak&quot; u Ruskom Krsturu, a po potrebi mogu biti održane i u lokalnoj biblioteci, prostorijama udruženja penzionera ili drugim lokalnim centrima."},
        {"role": "developer", 
         "content": "Koja je cena usluga?"},
        {"role": "assistant", 
         "content": "Cena pojedinačnih konsultacija kreće se od 200 do 1.000 RSD, u zavisnosti od vrste usluge. Cena grupnih obuka varira od 500 do 1.500 RSD po osobi, u zavisnosti od broja polaznika, sadržaja i trajanja obuke."},
        {"role": "developer", 
         "content": "Ko su članovi DIGIMOST-a?"},
        {"role": "assistant", 
         "content": "Članovi DIGIMOST-a su: Simona Miljanić – direktor, Sara Rajić – menadžer finansija i prodaje, Kalina Knežić – menadžer marketinga i prodaje"}
        # Replace with custom knowledge base.
    ]


def get_user_input() -> str:
    """Get user input from the command line."""
    return input("User: ")


def add_message(messages: list, role: str, content: str):
    """Add a message to the list of chat messages."""
    messages.append({"role": role, "content": content})


def generate_chat_response(messages: list) -> str:
    """Generate a chat response using the OpenAI API."""
    completion = client.chat.completions.create(model="gpt-4o-mini",
    messages=messages)
    return completion.choices[0].message.content


def main():
    messages = initialize_messages()

    while True:
        user_message = get_user_input()
        add_message(messages, "user", user_message)

        chat_response = generate_chat_response(messages)
        print(f'ChatGPT: {chat_response}')
        add_message(messages, "assistant", chat_response)


if __name__ == '__main__':
    main()
