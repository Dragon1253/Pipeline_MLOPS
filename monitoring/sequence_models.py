"""Étape 2.3 : modèles séquentiels (GRU, LSTM) pour prédire l'alerte.

Au lieu de tendances fabriquées à la main, chaque fenêtre ``t`` est décrite par la
séquence des ``length`` dernières fenêtres de son flux (``t-length+1 … t``) ; le réseau
apprend lui-même les évolutions. Entrées : ``sig_*``, ``known_*`` et ``label_delay``
seulement, plus un canal qui marque les positions de remplissage en début de flux.

Protocole identique à ``prediction.py`` : apprentissage sur ``train``, choix des
hyperparamètres et arrêt précoce sur ``validation`` (PR-AUC), seuil d'alarme fixé par le
même budget de fausses alarmes, mesure unique sur ``test`` et ``generalization``.

PyTorch est requis uniquement pour ce module (``requirements-monitoring.txt``).
"""
import numpy as np
from sklearn.metrics import average_precision_score

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - message explicite à l'utilisation
    torch = nn = None

SEQUENCE_LENGTH = 6
GRID = [{"hidden": 32}, {"hidden": 64}]


def require_torch():
    if torch is None:
        raise ImportError("PyTorch est nécessaire pour GRU/LSTM : "
                          "pip install -r requirements-monitoring.txt")


class Preprocessor:
    """Imputation par la médiane et standardisation, apprises sur l'entraînement seul."""

    def fit(self, frame, columns):
        values = frame[columns].to_numpy(dtype=float)
        self.columns = list(columns)
        self.median = np.nanmedian(values, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(values), values, self.median)
        self.mean, self.std = filled.mean(axis=0), filled.std(axis=0)
        self.std = np.where(self.std > 1e-9, self.std, 1.0)
        return self

    def transform(self, frame):
        values = frame[self.columns].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.median)
        missing = ~np.isfinite(frame[self.columns].to_numpy(dtype=float))
        scaled = (values - self.mean) / self.std
        # Indicateur de valeur manquante pour les métriques pas encore connues en début de flux.
        known = [i for i, c in enumerate(self.columns) if c.startswith("known_")]
        return np.concatenate([scaled, missing[:, known].any(axis=1, keepdims=True).astype(float)], axis=1)


def build_sequences(frame, matrix, length=SEQUENCE_LENGTH):
    """Tenseur (n_lignes, length, n_entrées + 1) aligné sur les lignes de ``frame``.

    La ligne ``t`` ne contient que les fenêtres ``<= t`` de son flux ; les positions avant
    le début du flux sont à zéro avec le dernier canal (remplissage) à 1.
    """
    n, width = matrix.shape
    sequences = np.zeros((n, length, width + 1), dtype=np.float32)
    sequences[:, :, -1] = 1.0
    stream_ids = frame["stream_id"].to_numpy()
    indexes = frame["index"].to_numpy()
    start = 0
    while start < n:
        end = start
        while end < n and stream_ids[end] == stream_ids[start]:
            end += 1
        if np.any(np.diff(indexes[start:end]) != 1):
            raise ValueError("Les lignes doivent être triées par flux puis par fenêtre.")
        for row in range(start, end):
            first = max(start, row - length + 1)
            count = row - first + 1
            sequences[row, length - count:, :width] = matrix[first:row + 1]
            sequences[row, length - count:, -1] = 0.0
        start = end
    return sequences


if nn is not None:
    class SequenceClassifier(nn.Module):
        def __init__(self, cell, n_inputs, hidden):
            super().__init__()
            recurrent = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
            self.recurrent = recurrent(n_inputs, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))

        def forward(self, x):
            output, _ = self.recurrent(x)
            return self.head(output[:, -1]).squeeze(-1)


def _predict(model, sequences, batch=4096):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            chunk = torch.from_numpy(sequences[start:start + batch])
            scores.append(torch.sigmoid(model(chunk)).numpy())
    return np.concatenate(scores)


def train_one(cell, params, train_x, train_y, val_x, val_y, *, epochs=30, patience=4, batch=256, seed=0):
    """Entraîne un réseau ; garde l'époque au meilleur PR-AUC de validation (arrêt précoce)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SequenceClassifier(cell, train_x.shape[2], params["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    positives = float(train_y.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - positives) / max(positives, 1e-6)))
    x_all, y_all = torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))
    best = (-1.0, None, 0)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch):
            idx = torch.from_numpy(order[start:start + batch])
            optimizer.zero_grad()
            loss = loss_fn(model(x_all[idx]), y_all[idx])
            loss.backward()
            optimizer.step()
        score = float(average_precision_score(val_y, _predict(model, val_x)))
        history.append({"epoch": epoch, "validation_pr_auc": score})
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "validation_pr_auc": best[0], "history": history}


class SequenceModel:
    """Enveloppe : prétraitement + séquences + réseau, avec ``predict_frame``."""

    def __init__(self, cell, preprocessor, network, length):
        self.cell, self.preprocessor, self.network, self.length = cell, preprocessor, network, length

    def predict_frame(self, frame):
        return _predict(self.network, build_sequences(frame, self.preprocessor.transform(frame), self.length))


def fit_sequence_models(train, validation, target, columns, cells=("gru", "lstm"), length=SEQUENCE_LENGTH,
                        grid=GRID, epochs=30, threads=None):
    """Entraîne GRU et LSTM ; hyperparamètres choisis sur la validation. ``train`` et ``validation``
    doivent être triés par flux puis par fenêtre et contenir toutes les fenêtres des flux."""
    require_torch()
    if threads:
        torch.set_num_threads(threads)
    preprocessor = Preprocessor().fit(train, columns)
    train_x = build_sequences(train, preprocessor.transform(train), length)
    val_x = build_sequences(validation, preprocessor.transform(validation), length)
    train_mask, val_mask = train[target].notna().to_numpy(), validation[target].notna().to_numpy()
    train_y = train[target].to_numpy()[train_mask]
    val_y = validation[target].to_numpy()[val_mask]
    models, tuning = {}, {}
    for cell in cells:
        results = []
        for params in grid:
            network, info = train_one(cell, params, train_x[train_mask], train_y, val_x[val_mask], val_y,
                                      epochs=epochs)
            results.append((info["validation_pr_auc"], params, network, info))
        best = max(results, key=lambda r: r[0])
        models[cell] = SequenceModel(cell, preprocessor, best[2], length)
        tuning[cell] = {"chosen": best[1], "sequence_length": length,
                        "runs": [{"params": p, **{k: v for k, v in i.items() if k != "history"},
                                  "history": i["history"]} for _, p, _, i in results]}
    return models, tuning
