import streamlit as st
import pandas as pd
import plotly.express as px
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
tab1, tab2 = st.tabs(["📈 Dashboard Geral", "🌍 Análise por País (Drill-Down)"])

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
    
    

