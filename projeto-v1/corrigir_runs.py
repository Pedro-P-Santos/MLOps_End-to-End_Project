import os
import yaml

def corrigir_run_uuid(mlruns_path):
    for exp_id in os.listdir(mlruns_path):
        exp_path = os.path.join(mlruns_path, exp_id)
        if not os.path.isdir(exp_path):
            continue

        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            meta_path = os.path.join(run_path, "meta.yaml")

            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)

                if "run_uuid" not in meta:
                    meta["run_uuid"] = run_id
                    with open(meta_path, "w") as f:
                        yaml.safe_dump(meta, f)
                    print(f"✔ Corrigido: {meta_path}")
                else:
                    print(f"✓ Já está ok: {meta_path}")

            except Exception as e:
                print(f"⚠ Erro a processar {meta_path}: {e}")

# Caminho relativo à raiz do projeto
caminho_mlruns = "mlruns"
corrigir_run_uuid(caminho_mlruns)
