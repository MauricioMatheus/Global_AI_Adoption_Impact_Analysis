import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
st.set_page_config(layout="wide", page_title="Dashboard de Adoçção de IA")

#Carregamento dos dados

@st.cache_data
def load_data():
    df = pd.read_csv("Global_AI_Content_Impact_Dataset.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df = load_data()

#Sidebar

st.sidebar.header("Filtros do Dashboard")

#Filtro por países (multiselect permite selecionar vários)
paises_disponiveis = sorted(df['country'].unique())
paises_selecionados = st.sidebar.multiselect(
    "Selecione os Países",
    options=paises_disponiveis,
    
    default=paises_disponiveis[:5] #Definindo os 5 primeiros países da lista como padrão
)
# Filtro por ano

ano_min = int(df['year'].min())
ano_max = int(df['year'].max())
anos_selecionados = st.sidebar.slider(
    "Selecione o período",
    min_value=ano_min,
    max_value=ano_max,
    value=(ano_min, ano_max) #Definindo o intervalo padrão como todos os anos
)

# criando um dataframe filtrado contendo apenas dados correspondentes às escolhas do usário
df_filtrado = df[
    (df['country'].isin(paises_selecionados)) &
    (df['year'].between(anos_selecionados[0], anos_selecionados[1]))
]

# título principal do Dashboard

st.title("🌎 Dashboard Interativo de Adoção de IA")
st.markdown("Use os filtros na barra lateral para explorar os dados na aba de dados gerais")

# Criando as duas abas principais da aplicação
tab1, tab2, tab3 = st.tabs(["📈 Dashboard Geral", "🌍 Análise por País", "🤖 Modelo Preditivo"])


# Aba 1: Dashboard geral

with tab1:
    st.header("Análise do Cenário Global e Regional")
    st.markdown("Use os filtros na barra lateral para explorar os dados agregados.")


    # Métricas principais

    col1, col2, col3 = st.columns(3)

    # Calculando as métricas com base nos dados filtrados
    media_adocao = df_filtrado['ai_adoption_rate_(%)'].mean()
    media_lucro = df_filtrado['revenue_increase_due_to_ai_(%)'].mean()
    media_perda_emprego = df_filtrado['job_loss_due_to_ai_(%)'].mean()

    col1.metric("Adoção Média de IA", f"{media_adocao:.2f}%")
    col2.metric("Aumento Médio de Lucro", f"{media_lucro:.2f}%")
    col3.metric("Média de Perda de Emprego", f"{media_perda_emprego:.2f}%")

    st.markdown("---") 

    # Gráficos do Dashboard

    # Mapa-Múndi interativo

    st.subheader("Taxa de Adoção de IA pelo Mundo")

    mapa_data = df.groupby('country')['ai_adoption_rate_(%)'].mean().reset_index()
    fig_mapa = px.choropleth(mapa_data,
                            locations="country",
                            locationmode="country names",
                            color="ai_adoption_rate_(%)",
                            hover_name="country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="Passe o mouse sobre um país para ver os detalhes")

    st.plotly_chart(fig_mapa, use_container_width=True) #Renderizando um gráfico na página

    col_graf1, col_graf2 = st.columns(2) #Layout com duas colunas para o próximo gráfico

    # Gráfico de barras na primeira coluna

    with col_graf1:
        st.subheader("Adoção de IA por Setor")
        adocao_setor = df_filtrado.groupby('industry')['ai_adoption_rate_(%)'].mean().sort_values(ascending=True).reset_index()
        fig_setor = px.bar(adocao_setor,
                        x='ai_adoption_rate_(%)',
                        y='industry',
                        orientation='h',
                        title="Média de Adoção de IA (%)",
                        text='ai_adoption_rate_(%)')
        fig_setor.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
        st.plotly_chart(fig_setor, use_container_width=True)
        
    # Gráfico de linha na segunda coluna

    with col_graf2:
        st.subheader("Tendência da Perda de Empregos")
        perda_emprego_ano = df_filtrado.groupby('year')['job_loss_due_to_ai_(%)'].mean().reset_index()
        fig_perda = px.line(perda_emprego_ano,
                            x='year',
                            y='job_loss_due_to_ai_(%)',
                            title="Média de Perda de Empregos com IA (%)",
                            markers=True)
        st.plotly_chart(fig_perda, use_container_width=True)
        
    # Gráfico de Bolhas Interativo

    st.subheader("Análise Multidimensional: Adoção vs. Lucro vs. Perda de Emprego")
    bubble_data = df_filtrado.groupby('country').agg({
        'ai_adoption_rate_(%)': 'mean',
        'revenue_increase_due_to_ai_(%)': 'mean',
        'job_loss_due_to_ai_(%)': 'mean'
    }).reset_index()

    fig_bubble = px.scatter(bubble_data,
                            x='ai_adoption_rate_(%)',
                            y='revenue_increase_due_to_ai_(%)',
                            size='job_loss_due_to_ai_(%)',
                            color='country',
                            hover_name='country',
                            size_max=50,
                            title="Análise por País (tamanho da bolha = perda de emprego)")
    st.plotly_chart(fig_bubble, use_container_width=True)
    
# Aba 2: Análise detalhada por país (Drill-Down)

with tab2:
    st.header("Análise Detalhada por País")
    st.markdown("Selecione um único país no menu abaixo para ver um relatório detalhado")
    
    # Seletor para escolher um único país para análise
    
    pais_unico = st.selectbox(
        "Selecione um País para Análise Aprofundada",
        options=paises_disponiveis,
        index=0 #Selecionando o primeiro país da lista
    )
    
    # Filtrando o df para o país selecionado
    df_pais = df[df['country'] == pais_unico]
    
    # Comparando o país selecionado com os dados globais
    st.subheader(f"Comparativo: {pais_unico} vs. Média Global")
    col_pais1, col_pais2 = st.columns(2)
    
    media_adocao_pais = df_pais['ai_adoption_rate_(%)'].mean()
    media_adocao_global = df['ai_adoption_rate_(%)'].mean()
    
    media_lucro_pais = df_pais['revenue_increase_due_to_ai_(%)'].mean()
    media_lucro_global = df['revenue_increase_due_to_ai_(%)'].mean()
    
    
    col_pais1.metric(
        f"Adoção Média em {pais_unico}",
        f"{media_adocao_pais:.2f}%",
        delta=f"{(media_adocao_pais - media_adocao_global):.2f}% vs. Média Global",
        delta_color="off"
    )
    
    col_pais2.metric(
        f"Lucro Médio com IA em {pais_unico}",
        f"{media_lucro_pais:.2f}%",
        delta=f"{(media_lucro_pais - media_lucro_global):.2f}% vs. Média Global",
        delta_color="off"
    )
    
    st.markdown("---")
    
    # Gráfico de linha de evolução dos indicadores no país selecionado
    st.subheader(f"Evolução dos Indicadores em {pais_unico} ao Longo dos Anos")
    
    df_pais_ano = df_pais.sort_values('year')
    
    fig_evolucao_pais = px.line(
        df_pais_ano,
        x='year',
        y=['ai_adoption_rate_(%)', 'revenue_increase_due_to_ai_(%)', 'job_loss_due_to_ai_(%)', 'consumer_trust_in_ai_(%)'],
        title=f"Tendências em {pais_unico}",
        markers=True,
        labels={'value': 'Valor (%)', 'variable': 'Métrica', 'year': 'Ano'}
    )
    st.plotly_chart(fig_evolucao_pais, use_container_width=True)
    
with tab3:
    st.header("🤖 Machine Learning Aplicado ao Negócio")
    st.markdown("Explore duas aplicações de IA: **Agrupamento** para segmentar países e **Regressão** para prever cenários.")
    st.markdown("---")

    # SEÇÃO 1: CLUSTERING 
    st.subheader("📊 Agrupamento dos Países por Nível de Adoção (Clustering)")
    st.markdown("Este modelo analisa a taxa de adoção de todos os países e os agrupa automaticamente em 3 categorias de performance.")
    cluster_data = df.groupby('country')['ai_adoption_rate_(%)'].mean().reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(cluster_data[['ai_adoption_rate_(%)']])
    cluster_centers = kmeans.cluster_centers_.flatten()
    order = cluster_centers.argsort()
    cluster_map = {order[0]: 'Baixa Adoção', order[1]: 'Média Adoção', order[2]: 'Alta Adoção'}
    cluster_data['Grupo de Adoção'] = cluster_data['cluster'].map(cluster_map)
    fig_cluster = px.bar(cluster_data.sort_values('ai_adoption_rate_(%)'),
                         x='country', y='ai_adoption_rate_(%)', color='Grupo de Adoção',
                         title="Classificação dos Países por Taxa Média de Adoção de IA",
                         labels={'ai_adoption_rate_(%)': 'Adoção Média (%)', 'country': 'País'}, height=600)
    st.plotly_chart(fig_cluster, use_container_width=True)
    for grupo in ['Alta Adoção', 'Média Adoção', 'Baixa Adoção']:
        st.write(f"**Países com {grupo}:**")
        lista_paises = cluster_data[cluster_data['Grupo de Adoção'] == grupo]['country'].sort_values().to_list()
        st.write(", ".join(lista_paises))

    st.markdown("---")

    # SEÇÃO 2: REGRESSÃO  
    st.subheader("📋 Previsão Pontual com Formulário (Regressão)")
    st.markdown("Preencha todos os campos e clique no botão para obter uma previsão específica.")
    df_ml = df.dropna().copy()
    le = LabelEncoder()
    df_ml['industry_encoded'] = le.fit_transform(df_ml['industry'])
    X = df_ml[['revenue_increase_due_to_ai_(%)', 'job_loss_due_to_ai_(%)', 'consumer_trust_in_ai_(%)', 'year', 'industry_encoded']]
    y = df_ml['ai_adoption_rate_(%)']
    modelo_form = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_form.fit(X, y)
    with st.form("form_predicao"):
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            ano = st.slider("Ano", int(df['year'].min()), int(df['year'].max()), 2023)
            lucro = st.slider("Aumento de lucro com IA (%)", 0.0, 100.0, 20.0)
            perda_emprego = st.slider("Perda de emprego com IA (%)", 0.0, 100.0, 10.0)
        with col_form2:
            confianca = st.slider("Confiança do consumidor em IA (%)", 0.0, 100.0, 50.0)
            setor = st.selectbox("Setor", sorted(df['industry'].dropna().unique()))
        btn = st.form_submit_button("Prever Adoção")
    if btn:
        setor_codificado = le.transform([setor])[0]
        entrada = np.array([[lucro, perda_emprego, confianca, ano, setor_codificado]])
        pred = modelo_form.predict(entrada)[0]
        st.success(f"A taxa prevista de adoção de IA para este cenário é de aproximadamente **{pred:.2f}%**")

    st.markdown("---")
    
    # SEÇÃO 3: SIMULADOR INTERATIVO - Random Forest
    st.subheader("🕹️ Simulador Interativo e Fatores de Influência (Regressão)")
    st.markdown("Ajuste os sliders e veja a previsão mudar em tempo real. O gráfico abaixo mostra quais fatores o modelo considera mais importantes.")
    @st.cache_resource
    def train_model_simulador():
        features = ['revenue_increase_due_to_ai_(%)', 'job_loss_due_to_ai_(%)', 'consumer_trust_in_ai_(%)']
        target = 'ai_adoption_rate_(%)'
        # Drop NaN para o modelo
        df_sim = df.dropna(subset=features + [target]).copy()
        X_sim = df_sim[features]
        y_sim = df_sim[target]
        model_sim = RandomForestRegressor(n_estimators=100, random_state=42)
        model_sim.fit(X_sim, y_sim)
        return model_sim
    model_simulador = train_model_simulador()
    
    # Adicionando a key em cada slider para evitar o erro "DuplicateWidgetID"
    input_revenue = st.slider("Simulador de Aumento do Lucro (%)", float(df['revenue_increase_due_to_ai_(%)'].min()), float(df['revenue_increase_due_to_ai_(%)'].max()), float(df['revenue_increase_due_to_ai_(%)'].mean()), key="sim_lucro")
    input_job_loss = st.slider("Simulador de Perda de Emprego (%)", float(df['job_loss_due_to_ai_(%)'].min()), float(df['job_loss_due_to_ai_(%)'].max()), float(df['job_loss_due_to_ai_(%)'].mean()), key="sim_perda")
    input_trust = st.slider("Simulador de Confiança do Consumidor (%)", float(df['consumer_trust_in_ai_(%)'].min()), float(df['consumer_trust_in_ai_(%)'].max()), float(df['consumer_trust_in_ai_(%)'].mean()), key="sim_confianca")
    
    prediction_sim = model_simulador.predict([[input_revenue, input_job_loss, input_trust]])
    st.metric("Taxa de Adoção de IA Prevista (Simulação em tempo real)", f"{prediction_sim[0]:.2f}%")
    
    st.write("**O que o modelo considera mais importante?**")
    importances = pd.DataFrame({
        'Feature': ['Aumento do Lucro', 'Perda de Emprego', 'Confiança do Consumidor'],
        'Importance': model_simulador.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig_importance = px.bar(importances, x='Importance', y='Feature', orientation='h', title='Importância de Cada Fator para a Previsão')
    st.plotly_chart(fig_importance, use_container_width=True)

    
    

