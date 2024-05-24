import os
from shiny import App, reactive, render, ui
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

directorioImagenes = "D:/Respaldo/Documents/Unach/8vo/Big Data/ProyectoFinal/output_images"
os.makedirs(directorioImagenes, exist_ok=True)

data = pd.read_csv("D:/Respaldo/Documents/Unach/8vo/Big Data/ProyectoFinal/WallCityTap_Consumer.csv")
estilos = "D:/Respaldo/Documents/Unach/8vo/Big Data/ProyectoFinal/estilos.css"

app_ui = ui.page_fluid(
    ui.include_css(estilos),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider("age_range", "Edad:", min=data['Age'].min(), max=data['Age'].max(), value=[data['Age'].min(), data['Age'].max()]),
            ui.input_checkbox_group("payment_methods", "Métodos de Pago:",
                                    choices=["Cash", "Tcredit", "Tdebit"],
                                    selected=["Cash", "Tcredit", "Tdebit"]),
            ui.input_slider("n_clusters", "Numero de clusters", min=2, max=10, value=6)
        ),
        ui.panel_main(
            ui.row(
                ui.column(4, ui.div(ui.output_text_verbatim("enLinea"), class_="stat-box")),
                ui.column(4, ui.div(ui.output_text_verbatim("medioPago"), class_="stat-box")),
                ui.column(4, ui.div(ui.output_text_verbatim("promedioEdad"), class_="stat-box"))
            ),
            ui.div(
                ui.h2("Segmentación: Medio de Pago"),
                ui.div(ui.output_image("clusterF"), class_="plot-container"),
                class_="centered-seg"
            ),
            ui.div(ui.h2("Conjunto de datos"), class_="centered-seg"),
            ui.div(ui.output_data_frame("tabla"), class_="table-container")
        )
    )
)

def server(input, output, session):
    @reactive.Calc
    def filtrado():
        df = data.copy()
        df = df[df['Age'].between(*input.age_range())]
        df = df[df['Payment_Methods'].isin(input.payment_methods())]
        return df
    
    @output
    @render.text
    def enLinea():
        df = filtrado()
        if 'OnlinePurchase' in df.columns:
            return f"Número de consumidores online E-commerce: {df['OnlinePurchase'].sum()}"
        else:
            return "Error: 'OnlinePurchase' no encontrado"

    @output
    @render.text
    def medioPago():
        df = filtrado()
        if not df[df['Payment_Methods'] == 'Cash'].empty:
            avg_age = df[df['Payment_Methods'] == 'Cash']['Age'].mean()
            return f"Edad promedio de consumidores que utilizan como medio de pago: {avg_age:.2f}"
        else:
            return "Edad promedio de consumidores que utilizan como medio de pago:"

    @output
    @render.text
    def promedioEdad():
        df = filtrado()
        df_high_income = df[df['Annual_Income'] > 20]
        if not df_high_income.empty:
            avg_age = df_high_income['Age'].mean()
            return f"Edad promedio de consumidores que poseen ingresos mayores $20,000: {avg_age:.2f}"
        else:
            return "Edad promedio de consumidores que poseen ingresos mayores $20,000:"
    
    @output
    @render.image
    def clusterF():
        df = filtrado()
        if df[['Age', 'Annual_Income']].empty:
            return None
        kmeans = KMeans(n_clusters=input.n_clusters(), random_state=0)
        df['Cluster'] = kmeans.fit_predict(df[['Age', 'Annual_Income']])
        fig = px.scatter(df, x='Annual_Income', y='Age', color='Cluster', symbol='Payment_Methods', title='Segmentación: Medio de Pago', color_continuous_scale=px.colors.sequential.Viridis)
        
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=1.2
        ))

        # Guardar la imagen en el directorio conocido
        imagenCluster = os.path.join(directorioImagenes, "cluster_plot.png")
        fig.write_image(imagenCluster)

        return {"src": imagenCluster, "alt": "Segmentación: Medio de Pago"}

    @output
    @render.data_frame
    def tabla():
        # Convertir los valores booleanos a texto para asegurar que se muestran
        df = filtrado().copy()
        df['OnlinePurchase'] = df['OnlinePurchase'].astype(str)
        return df

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8080)
