"""
Interface Gradio — Classification de Revues Scientifiques
Modèle : BertForMultiLabelClassification (bert_multilabel_best.pth)
Multi-label : une revue peut appartenir à plusieurs classes simultanément
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import BertForMultiLabelClassification   # votre fichier model.py

# Configuration 

MODEL_NAME  = "bert-base-uncased"
#MODEL_PATH  = "best_bert_multilabel.pth"
MODEL_PATH  = "C:\\Users\\iboug\\Desktop\\bert-classification-scientific publication\\bert_multilabel_best_model .pt" # test du model entrainer avec la weight_ponderation sans sampler
MAX_LENGTH  = 256
THRESHOLD   = 0.5

CLASSES = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Chargement du modèle (une seule fois au démarrage) 

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMultiLabelClassification(MODEL_NAME, n_class=6)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

print("Chargement du modèle…")
MODEL, TOKENIZER = load_model()
print(f"Modèle chargé sur {DEVICE} ✓")

#Prédiction 

def predict(title: str, abstract: str, threshold: float):
    if not title.strip() and not abstract.strip():
        raise gr.Error("Veuillez saisir au moins le titre ou le résumé.")

    # Concaténation titre + abstract (même logique que l'entraînement)
    text = f"{title} [SEP] {abstract}"

    encoding = TOKENIZER(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = MODEL(input_ids, attention_mask=attention_mask)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Scores pour gr.Label (toutes les classes)
    scores = {cls: float(round(float(p), 4)) for cls, p in zip(CLASSES, probs)}

    # Classes retenues au-dessus du seuil
    predicted = [cls for cls, p in zip(CLASSES, probs) if p >= threshold]
    label_str  = ", ".join(predicted) if predicted else "Aucune classe au-dessus du seuil"

    return scores, label_str


# Exemples 

EXAMPLES = [
    [
        "Attention Is All You Need",
        "We propose a new simple network architecture, the Transformer, based solely on "
        "attention mechanisms. Experiments on two machine translation tasks show these models "
        "to be superior in quality while being more parallelizable.",
        0.5,
    ],
    [
        "Gravitational Waves from Binary Black Hole Mergers",
        "We report the observation of gravitational waves from the merger of two stellar-mass "
        "black holes, consistent with predictions from general relativity.",
        0.5,
    ],
    [
        "The Riemann Hypothesis and Distribution of Prime Numbers",
        "We study the connection between the non-trivial zeros of the Riemann zeta function "
        "and the distribution of prime numbers using spectral methods on L-functions.",
        0.5,
    ],
    [
        "Bayesian Estimation of Stochastic Volatility Models",
        "We develop a fully Bayesian approach to the estimation of stochastic volatility "
        "models for financial time series using Markov chain Monte Carlo methods.",
        0.5,
    ],
    [
        "Protein Folding Prediction Using Deep Learning",
        "We present a deep learning architecture for predicting three-dimensional protein "
        "structure from amino acid sequence with near-experimental accuracy.",
        0.5,
    ],
    [
        "Option Pricing Under Jump-Diffusion Processes",
        "We derive closed-form approximations for European option prices under a "
        "jump-diffusion model with stochastic volatility via Fourier transform methods.",
        0.5,
    ],
]

# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="ArXiv Paper Classifier") as demo:

    gr.Markdown("# ArXiv Paper Classifier")
    gr.Markdown(
        "Classification **multi-label** : un article peut appartenir à plusieurs domaines simultanément.  \n"
        "Modèle : `bert-base-uncased` fine-tuné sur des revues ArXiv."
    )

    with gr.Row():

        # ── Colonne entrées ───────────────────────────────────────────────────
        with gr.Column():
            title_input = gr.Textbox(
                label="Titre de l'article",
                placeholder="Ex : Attention Is All You Need",
                lines=2,
            )
            abstract_input = gr.Textbox(
                label="Résumé / Abstract",
                placeholder="Collez ici le résumé de l'article…",
                lines=10,
            )
            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Seuil de décision (sigmoid > seuil → classe activée)",
            )
            with gr.Row():
                clear_btn  = gr.ClearButton(
                    components=[title_input, abstract_input],
                    value="Effacer",
                )
                submit_btn = gr.Button("Classifier", variant="primary")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[title_input, abstract_input, threshold_slider],
                label="Exemples d'articles",
            )

        # ── Colonne résultats ─────────────────────────────────────────────────
        with gr.Column():
            output_label = gr.Label(
                label="Probabilités par classe (sigmoid)",
                num_top_classes=6,
            )
            output_classes = gr.Textbox(
                label="Classes prédites (au-dessus du seuil)",
                interactive=False,
            )

    # ── Événements 
    submit_btn.click(
        fn=predict,
        inputs=[title_input, abstract_input, threshold_slider],
        outputs=[output_label, output_classes],
        api_name="classify",
    )
    abstract_input.submit(
        fn=predict,
        inputs=[title_input, abstract_input, threshold_slider],
        outputs=[output_label, output_classes],
    )
    threshold_slider.change(
        fn=predict,
        inputs=[title_input, abstract_input, threshold_slider],
        outputs=[output_label, output_classes],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=True,  # True pour un lien public temporaire
    )