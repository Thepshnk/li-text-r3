import numpy as np
import pickle
import os
import re
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class LiTextR3:
    def __init__(self, knowledge_file="base.txt"):
        self.knowledge_file = knowledge_file
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.knowledge_vectors = None
        self.conversation_history = []
        
        self.load_and_train()
        
    def load_and_train(self):
        print("Загрузка базы знаний...")
        
        if not os.path.exists(self.knowledge_file):
            print(f"Файл {self.knowledge_file} не найден!")
            return
            
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sentences = nltk.sent_tokenize(content)
        self.knowledge_base = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        print(f"Загружено {len(self.knowledge_base)} предложений")
        
        if self.knowledge_base:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
            print("Модель обучена на вашей базе знаний!")
        else:
            print("В базе знаний недостаточно данных!")
    
    def find_best_match(self, query):
        if self.knowledge_vectors is None:
            return "База знаний еще не обучена"
            
        query_vec = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vec, self.knowledge_vectors)
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0, best_match_idx]
        
        if best_similarity > 0.1:
            return self.knowledge_base[best_match_idx]
        else:
            return "В моей базе знаний нет информации по этому вопросу"
    
    def learn_new_fact(self, fact):
        if fact not in self.knowledge_base:
            self.knowledge_base.append(fact)
            
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
            
            with open(self.knowledge_file, 'a', encoding='utf-8') as f:
                f.write(fact + '\n')
            
            return f"Выучил новый факт: {fact}"
        else:
            return "Этот факт уже есть в базе знаний"
    
    def process_query(self, user_input):
        user_input = user_input.lower().strip()
        
        self.conversation_history.append(user_input)
        
        if user_input.startswith('запомни:'):
            fact = user_input.replace('запомни:', '').strip()
            return self.learn_new_fact(fact)
            
        elif user_input.startswith('обучись:'):
            fact = user_input.replace('обучись:', '').strip()
            return self.learn_new_fact(fact)
        
        response = self.find_best_match(user_input)
        
        if response == "В моей базе знаний нет информации по этому вопросу":
            if any(word in user_input for word in ['привет', 'здравствуй', 'hello', 'hi']):
                return "Привет! Я li text r3. Спроси меня о чем-то из моей базы знаний или научи чему-то новому командой 'запомни: твой факт'"
            elif any(word in user_input for word in ['пока', 'выход', 'exit']):
                return "До свидания! Буду рад пообщаться снова."
        
        return response
    
    def save_model(self, filename="li_text_r3_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'knowledge_base': self.knowledge_base,
                'vectorizer': self.vectorizer,
                'knowledge_vectors': self.knowledge_vectors
            }, f)
        print(f"Модель сохранена в {filename}")
    
    def load_model(self, filename="li_text_r3_model.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.knowledge_base = data['knowledge_base']
            self.vectorizer = data['vectorizer']
            self.knowledge_vectors = data['knowledge_vectors']
            print(f"Модель загружена из {filename}")

def main():
    bot = LiTextR3("base.txt")
    
    print("=" * 50)
    print("Li Text R3 - Автономная языковая модель")
    print("Обучается ТОЛЬКО на вашей базе base.txt")
    print("Команды: 'запомни: факт' или 'обучись: факт'")
    print("Для выхода введите 'выход'")
    print("=" * 50)
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            bot.save_model()
            print("Модель сохранена. До свидания!")
            break
            
        response = bot.process_query(user_input)
        print(f"Бот: {response}")

if __name__ == "__main__":
    main()
