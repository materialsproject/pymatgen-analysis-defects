import click
import dash
import numpy as np
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from monty.serialization import loadfn
from pymatgen.core import Element


@click.command()
@click.argument("filename", type=click.Path(exists=True))
def main(filename: str) -> None:
    fed = loadfn(filename)

    def chempot_plot(bulk_formula):
        """
        Draw the chemical potentia plot
        """
        cpd = fed.chempot_diagram
        def_set = set(fed.chempot_diagram.elements) - {Element("Ga"), Element("O")}
        def_el = next(iter(def_set))
        cpd.limits = {def_el: [-4, 0]}
        cpd = cpd.from_dict(cpd.as_dict())

        fig = cpd.get_plot(formulas_to_draw=[bulk_formula])
        fig.update_traces(hoverinfo="skip")
        x, y, z = cpd.domains[bulk_formula].T
        fig.add_scatter3d(x=x, y=y, z=z, mode="markers")
        return fig

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        className="row",
        children=[
            html.H1(
                "Select chemical potential limit to obtain the formation energy diagram."
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id="chempot",
                        figure=chempot_plot("Ga2O3"),
                        style={"display": "inline-block"},
                    ),
                    dcc.Graph(id="formation-en", style={"display": "inline-block"}),
                ]
            ),
        ],
    )

    @app.callback(Output("formation-en", "figure"), Input("chempot", "clickData"))
    def select_chempot(clickData):
        if clickData is None:
            form_en_plot = px.line(
                x=[
                    0,
                ],
                y=[0],
            )
        else:
            cp_point = clickData["points"][0]
            chempot = [cp_point["x"], cp_point["y"], cp_point["z"]]
            cp_dict = fed._parse_chempots(chempot)
            form_en = np.array(fed.get_transitions(cp_dict, 0, 5))

            form_en_plot = px.line(x=form_en[:, 0], y=form_en[:, 1])
        form_en_plot.update_yaxes(range=[-3, 8])
        custom_layout = {
            "width": 500,
            "height": 700,
            "hovermode": "closest",
            "paper_bgcolor": "rgba(256,256,256,100)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis_title": "Fermi level (eV)",
            "yaxis_title": "Formation energy (eV)",
            "showlegend": True,
            "legend": {
                "orientation": "v",
                "x": 0.1,
                "y": 0.99,
                "traceorder": "reversed",
                "xanchor": "left",
                "yanchor": "top",
            },
            "xaxis": {"gridcolor": "#dbdbdb", "gridwidth": 2, "showline": True},
            "yaxis": {"gridcolor": "#dbdbdb", "gridwidth": 2, "showline": True},
        }
        form_en_plot.update_layout(custom_layout)
        return form_en_plot

    app.run_server(debug=True, use_reloader=True)


if __name__ == "__main__":
    main()
