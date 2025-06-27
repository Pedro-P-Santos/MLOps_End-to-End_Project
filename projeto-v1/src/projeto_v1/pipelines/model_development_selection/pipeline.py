# from kedro.pipeline import Pipeline, node
# from .nodes import model_selection


# def create_pipeline(**kwargs):
#     return Pipeline([
#         node(
#             func=model_selection,
#             inputs=[
#                 "X_train_preprocessed", "X_test_preprocessed", "y_train_encoded", "y_test_encoded",
#                 "parameters_model_selection", "parameters_grid", "final_selected_features",
#             ],
#             outputs="champion_model",
#             name="model_selection_node"
#         )
#     ])



from kedro.pipeline import Pipeline, node
from .nodes import model_selection

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=model_selection,
            inputs=[
                "X_train_preprocessed",
                "X_test_preprocessed",
                "y_train_encoded",
                "y_test_encoded",
                "params:parameters_model_selection",
                "params:parameters_grid",
                "final_selected_features"
            ],
            outputs=[
                "champion_model",
                "X_train_scaled",
                "X_test_scaled"
            ],
            name="model_selection_node"
        )
    ])

