"""
scripts/setup_cronjob.py
────────────────────────
Registra (ou atualiza) o keep-alive do Render/Neon no cron-job.org via API.

Uso (uma única vez):
    CRONJOB_API_KEY=<chave> python scripts/setup_cronjob.py
    CRONJOB_API_KEY=<chave> python scripts/setup_cronjob.py --interval 5
    CRONJOB_API_KEY=<chave> python scripts/setup_cronjob.py --list
    CRONJOB_API_KEY=<chave> python scripts/setup_cronjob.py --delete <job_id>

Como obter a chave:
    cron-job.org → Login → Menu superior → API (canto direito)

Documentação da API:
    https://docs.cron-job.org/rest-api.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

API_BASE    = "https://api.cron-job.org"
TARGET_URL  = "https://vies-detector.onrender.com/api/warmup"
JOB_TITLE   = "BiasRadar — keep-alive Render + Neon"


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _minutes_for_interval(interval: int) -> list[int]:
    """Retorna a lista de minutos em que o job deve disparar."""
    return list(range(0, 60, interval))


def create_job(api_key: str, interval: int) -> None:
    minutes = _minutes_for_interval(interval)

    payload = {
        "job": {
            "url": TARGET_URL,
            "title": JOB_TITLE,
            "enabled": True,
            "saveResponses": True,       # guarda último response no dashboard
            "schedule": {
                "timezone": "UTC",
                "expiresAt": 0,          # sem expiração
                "hours":  [-1],          # toda hora
                "mdays":  [-1],          # todo dia do mês
                "minutes": minutes,
                "months": [-1],          # todo mês
                "wdays":  [-1],          # todo dia da semana
            },
            "requestTimeout": 30,
            "redirectSuccess": True,
            "notification": {
                "onFailure": True,       # alerta por e-mail se a API retornar erro
                "onSuccess": False,
                "onDisable": True,
            },
        }
    }

    resp = requests.put(
        f"{API_BASE}/jobs",
        headers=_headers(api_key),
        json=payload,
        timeout=15,
    )

    if not resp.ok:
        print(f"Erro {resp.status_code}: {resp.text}")
        sys.exit(1)

    job_id = resp.json().get("jobId")
    print(f"Keep-alive criado com sucesso!")
    print(f"  Job ID   : {job_id}")
    print(f"  URL      : {TARGET_URL}")
    print(f"  Intervalo: a cada {interval} minutos ({minutes})")
    print(f"  Dashboard: https://cron-job.org/en/members/jobs/edit/{job_id}/")


def list_jobs(api_key: str) -> None:
    resp = requests.get(f"{API_BASE}/jobs", headers=_headers(api_key), timeout=15)

    if not resp.ok:
        print(f"Erro {resp.status_code}: {resp.text}")
        sys.exit(1)

    jobs = resp.json().get("jobs", [])
    if not jobs:
        print("Nenhum job encontrado.")
        return

    print(f"{'ID':>10}  {'Enabled':>7}  {'URL'}")
    print("-" * 70)
    for j in jobs:
        enabled = "sim" if j.get("enabled") else "não"
        print(f"{j['jobId']:>10}  {enabled:>7}  {j.get('url', '')}")


def delete_job(api_key: str, job_id: int) -> None:
    resp = requests.delete(
        f"{API_BASE}/jobs/{job_id}",
        headers=_headers(api_key),
        timeout=15,
    )

    if not resp.ok:
        print(f"Erro {resp.status_code}: {resp.text}")
        sys.exit(1)

    print(f"Job {job_id} removido com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerencia keep-alive no cron-job.org")
    parser.add_argument("--interval", type=int, default=3,
                        help="Intervalo em minutos (padrão: 3, mín recomendado: 3)")
    parser.add_argument("--list",   action="store_true", help="Lista jobs existentes")
    parser.add_argument("--delete", type=int, metavar="JOB_ID", help="Remove um job pelo ID")
    args = parser.parse_args()

    api_key = os.getenv("CRONJOB_API_KEY")
    if not api_key:
        print("Erro: variável CRONJOB_API_KEY não definida.")
        print("Obtenha em: cron-job.org → Login → API (menu superior)")
        sys.exit(1)

    if args.list:
        list_jobs(api_key)
    elif args.delete:
        delete_job(api_key, args.delete)
    else:
        create_job(api_key, args.interval)


if __name__ == "__main__":
    main()
