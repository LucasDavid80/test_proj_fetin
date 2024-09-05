from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Carregar o modelo treinado e o LabelEncoder
with open("modelo_knn.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    convertVar = pickle.load(f)

# Perguntas baseadas no dataset
conversas = [
    "Olá, sou Maind sua IA para doenças mentais, pode me responder algumas perguntas?",
    "Você está se sentindo nervoso?",
    "Você está tendo ataques de pânico?",
    "Sua respiração está rápida?",
    "Você está suando?",
    "Está tendo problemas para se concentrar?",
    "Está tendo dificuldades para dormir?",
    "Está tendo problemas no trabalho?",
    "Você se sente sem esperança?",
    "Você está com raiva?",
    "Você tende a exagerar?",
    "Você percebe mudanças nos seus hábitos alimentares?",
    "Você tem pensamentos suicidas?",
    "Você se sente cansado?",
    "Você tem um amigo próximo?",
    "Você tem vício em redes sociais?",
    "Você ganhou peso recentemente?",
    "Você valoriza muito as posses materiais?",
    "Você se considera introvertido?",
    "Lembranças estressantes estão surgindo?",
    "Você tem pesadelos?",
    "Você evita pessoas ou atividades?",
    "Você está se sentindo negativo?",
    "Está com problemas de concentração?",
    "Você tende a se culpar por coisas?"
]

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas as rotas

# Estado da conversa
estado_conversa = {
    'indice_pergunta': 0,
    'respostas': []
}


@app.route('/mAInd', methods=['POST'])
def receive_text():
    data = request.get_json()

    text_mensage = data.get('text_mensage')

    if text_mensage == 'sim':
        estado_conversa["respostas"].append(1)
        estado_conversa["indice_pergunta"] += 1
    elif text_mensage == 'não':
        estado_conversa["respostas"].append(0)
        estado_conversa["indice_pergunta"] += 1

    # Verifica se todas as perguntas foram feitas
    if estado_conversa["indice_pergunta"] < len(conversas):
        response_text = f"{conversas[estado_conversa['indice_pergunta']]} (sim/não): "
    else:
        # Se todas as perguntas foram respondidas, chama a função coletarRespostas
        response_text = coletar_respostas(estado_conversa["respostas"])

        # Reseta o estado da conversa se necessário
        estado_conversa["indice_pergunta"] = 0
        estado_conversa["respostas"] = []

    return jsonify({"response_text": response_text}), 200


def coletar_respostas(respostas):
    # Eliminando a primeira resposta pois não é necessária no treinamento
    respostas.pop(0)

    # Fazer a predição com base nas respostas do usuário
    predicao = modelo.predict([respostas])

    # Converter a predição de volta para o transtorno correspondente
    transtorno_predito = convertVar.inverse_transform(predicao)[0]

    # Diagnósticos com base no transtorno predito
    diagnosticos = {
        'Normal': "Parece que você está bem no momento. Continue cuidando de si mesmo!",
        'Stress': "Parece que você está passando por um período de estresse. Não hesite em tirar um tempo para si ou buscar ajuda se sentir que precisa.",
        'Loneliness': "Pode ser que você esteja se sentindo um pouco sozinho agora. Lembre-se de que conectar-se com alguém pode fazer a diferença.",
        'Depression': "As suas respostas sugerem que você pode estar enfrentando sintomas de depressão. Falar com um profissional pode ajudar a entender melhor o que está acontecendo.",
        'Anxiety': "Você parece estar apresentando sinais de ansiedade. Seria bom considerar falar com alguém para entender melhor esses sentimentos."
    }

    return diagnosticos.get(transtorno_predito, "Erro ao identificar transtorno.")


if __name__ == '__main__':
    app.run(debug=True)
