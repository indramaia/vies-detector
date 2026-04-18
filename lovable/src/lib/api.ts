/**
 * src/lib/api.ts
 * Cliente centralizado para a API Flask do vies_detector.
 * Cole este arquivo no editor do Lovable em src/lib/api.ts
 */

// ── Tipos (espelham os modelos Python) ────────────────────────────────────────

export type BiasInterpretation =
  | "Predominantemente factual"
  | "Viés moderado"
  | "Viés elevado"
  | "Linguagem fortemente enviesada";

export type PositionLabel =
  | "Esquerda"
  | "Centro-progressista"
  | "Centro"
  | "Centro-conservador"
  | "Direita";

export interface VehicleIndex {
  ideology_id: string;
  source_name: string;
  computed_at: string;           // ISO 8601
  window_days: number;
  article_count: number;
  mean_bias: number;             // [0.0, 2.0]
  ideology_score: number | null; // [-1.0, 1.0]
  uncertainty: number | null;
  position_label: PositionLabel | null;
  contextualization: string;
  caveat: string;
}

export interface SpectrumEntry {
  source_name: string;
  ideology_id: string;
  ideology_score: number | null;
  uncertainty: number | null;
  position_label: PositionLabel | null;
  bias_score: number;
  article_count: number;
}

export interface Article {
  url_hash: string;
  title: string;
  url: string;
  source_name: string;
  published_at: string;          // ISO 8601
  bias_score: number;            // [0.0, 2.0]
  bias_interpretation: BiasInterpretation;
  sentence_count: number;
  n_factual: number;
  n_biased: number;
  n_strongly_biased: number;
}

export interface HealthStatus {
  status: "ok" | "error";
  timestamp: string;
}

// ── Cliente base ──────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? "";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new ApiError(res.status, body.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

// ── Funções de acesso aos endpoints ──────────────────────────────────────────

export const api = {
  /** GET /api/health */
  health: () => get<HealthStatus>("/api/health"),

  /** GET /api/vehicles — todos os veículos */
  vehicles: () => get<VehicleIndex[]>("/api/vehicles"),

  /** GET /api/vehicles/:id — um veículo específico */
  vehicle: (ideologyId: string) =>
    get<VehicleIndex>(`/api/vehicles/${ideologyId}`),

  /** GET /api/spectrum — veículos ordenados no espectro ideológico */
  spectrum: () => get<SpectrumEntry[]>("/api/spectrum"),

  /**
   * GET /api/articles?source=:id&limit=:n
   * @param ideologyId  ideology_id do veículo (obrigatório)
   * @param limit       máximo de artigos (padrão 20, máx 100)
   */
  articles: (ideologyId: string, limit = 20) =>
    get<Article[]>(`/api/articles?source=${ideologyId}&limit=${limit}`),
};

// ── Helpers de UI ─────────────────────────────────────────────────────────────

/** Converte mean_bias [0,2] em percentuais { factual, biased, strongly } */
export function biasToPercents(vehicle: VehicleIndex | Article) {
  const total =
    "n_factual" in vehicle
      ? vehicle.sentence_count
      : vehicle.article_count;

  if ("n_factual" in vehicle && vehicle.sentence_count > 0) {
    return {
      factual: Math.round((vehicle.n_factual / vehicle.sentence_count) * 100),
      biased: Math.round((vehicle.n_biased / vehicle.sentence_count) * 100),
      strongly: Math.round(
        (vehicle.n_strongly_biased / vehicle.sentence_count) * 100,
      ),
    };
  }

  // Para VehicleIndex sem contagens individuais, estima pela fórmula inversa
  const score = "mean_bias" in vehicle ? vehicle.mean_bias : vehicle.bias_score;
  const strongly = Math.round((score / 2) * 40);
  const biased = Math.round((score / 2) * 30);
  const factual = 100 - strongly - biased;
  return { factual: Math.max(factual, 0), biased, strongly };
}

/** Cor do badge por position_label */
export const IDEOLOGY_COLORS: Record<string, string> = {
  Esquerda: "#EF4444",
  "Centro-progressista": "#F97316",
  Centro: "#22C55E",
  "Centro-conservador": "#3B82F6",
  Direita: "#6366F1",
};

/** Cor da barra por faixa de bias_score */
export function biasColor(score: number): string {
  if (score < 0.4) return "#22C55E";
  if (score < 0.8) return "#F59E0B";
  if (score < 1.4) return "#F97316";
  return "#EF4444";
}

/** Rótulo por faixa de bias_score */
export function biasLabel(score: number): BiasInterpretation {
  if (score < 0.4) return "Predominantemente factual";
  if (score < 0.8) return "Viés moderado";
  if (score < 1.4) return "Viés elevado";
  return "Linguagem fortemente enviesada";
}
