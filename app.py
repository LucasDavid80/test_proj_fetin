from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
# import gdown
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# # substitua "id_do_arquivo" pelo id do arquivo no Google Drive
# url = "https://drive.google.com/uc?id=1C0A11bIv9i_SIxADjv0zg4eiMlvpIx1j"
# output = "dataset.csv"
# gdown.download(url, output, quiet=False)

# # carregar o dataset CSV
# data = pd.read_csv(output)

# carregar os dados CSV
chunks = pd.read_csv(r"dataset.csv", chunksize=1000)
data = pd.concat(chunks, ignore_index=True)

'''---------------------------TREINAMENTO----------------------------------------------------------'''
# converter valores categóricos em valores numéricos (vai ser usado no KNN)
convertVar = LabelEncoder()

# aplica LabelEncoder em todas as colunas categóricas de X
for column in data.columns[:-1]:  # Exclui a última coluna 'Disorder'
    data[column] = convertVar.fit_transform(data[column])

# codificar a variável target (Disorder)
data['Disorder'] = convertVar.fit_transform(data['Disorder'])

# dividir os dados em features (X) e rótulos (y)
# Todas as colunas, exceto a última (entrada)
X = data.drop(columns=['Disorder'])
Y = data['Disorder']  # A última coluna (saída)

# divide os dados em conjunto de treino e teste
xTreino, xTeste, yTreino, yTeste = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# inicializar o modelo
modelo = KNeighborsClassifier()

# treinar o modelo
modelo.fit(xTreino, yTreino)

# avaliar o desempenho do modelo
precisao = modelo.score(xTeste, yTeste)
print("Precisão do modelo(KNN): {}".format(precisao))

'''-------------------------------CHATBOT-----------------------------------'''
# perguntas baseadas nas colunas do dataset
conversas = [
    "Olá, sou Maind sua ia para doenças mentais, pode me responder algumas perguntas (Lembrando que eu não substituo um profissional)?",
    "Voce está se sentindo nervoso?",
    "Voce está tendo ataques de panico?",
    "Sua respiração está rápida?",
    "Voce está suando?",
    "Está tendo problemas para se concentrar?",
    "Está tendo dificuldades para dormir",
    "Está tendo problemas no trabalho?",
    "Você se sente sem esperança?",
    "Você esta com raiva?",
    "Você tende a exagerar?",
    "Você percebe mudanças nos seus habitos alimentares?",
    "Você tem pensamentos suicidas?",
    "Você se sente cansado?",
    "Você tem um amigo proximo?",
    "Você tem vicio em redes sociais?",
    "Você ganhou peso recentemente?",
    "Você valoriza muito as posses materiais?",
    "Você se considera introvertido?",
    "Lembranças estressantes estão surgindo?",
    "Você tem pesadelos?",
    "Você evita pessoas ou atividades?",
    "Você está se sentindo negativo?",
    "Está com problemas de concentraçao?",
    "Você tende a se culpar por coisas?"
]

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

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
        estado_conversa["indice_pergunta"] = estado_conversa["indice_pergunta"] + 1
    elif text_mensage == 'não':
        estado_conversa["respostas"].append(0)
        estado_conversa["indice_pergunta"] = estado_conversa["indice_pergunta"] + 1

    if estado_conversa["indice_pergunta"] < 24:
        response_text = f"{conversas[estado_conversa['indice_pergunta']]} (sim/não): "
    else:
        response_text = coletarRespostas(estado_conversa["respostas"])

    if estado_conversa["indice_pergunta"] == "Você tende a se culpar por coisas?":
        estado_conversa["indice_pergunta"] = 0

    return jsonify({"response_text": response_text}), 200


def coletarRespostas(respostas):

    # fazer a predição com base nas respostas do usuário
    predicao = modelo.predict([respostas])

    # converter a predição de volta para o transtorno correspondente
    transtornoPredito = convertVar.inverse_transform(predicao)[0]

    # Devolve com o transtorno
    # diagnosticos
    if transtornoPredito == 'Normal':
        return ("Com base nas suas respostas, você pode ta suave meu parceiro")

    if transtornoPredito == 'Stress':
        return ("Com base nas informações dadas voce pode estar com estresse, lembresse de procurar um profissional especializado para ter certeza!")

    if transtornoPredito == 'Loneliness':
        return ("Com base nas informacoes voce pode estar se sentindo solitário!")

    if transtornoPredito == 'Depression':
        return ("Com base nas informaçoes que voce forneceu, voce indica sintomas de depressão, procure um profissional para ter certeza que está tudo bem!")

    if transtornoPredito == 'Anxiety':
        return ("Com base nas informações que voce forneceu você parece estar com ansiedade, procure um profissional para ter certeza que está tudo bem!")


if __name__ == '__main__':
    app.run()
