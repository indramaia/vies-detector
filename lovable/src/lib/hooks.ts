/**
 * src/lib/hooks.ts
 * Hooks TanStack Query para todos os endpoints do vies_detector.
 * Cole este arquivo no editor do Lovable em src/lib/hooks.ts
 *
 * Pré-requisito no projeto Lovable:
 *   npm install @tanstack/react-query
 *
 * Em main.tsx envolva o app com:
 *   <QueryClientProvider client={new QueryClient()}>
 *     <App />
 *   </QueryClientProvider>
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

// ── Chaves de cache ───────────────────────────────────────────────────────────

export const queryKeys = {
  health: ["health"] as const,
  vehicles: ["vehicles"] as const,
  vehicle: (id: string) => ["vehicles", id] as const,
  spectrum: ["spectrum"] as const,
  articles: (id: string, limit?: number) => ["articles", id, limit] as const,
};

// ── Hooks ─────────────────────────────────────────────────────────────────────

/** Status da API — usado no header/status bar */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.health,
    refetchInterval: 60_000, // verifica a cada 1 min
    staleTime: 30_000,
  });
}

/**
 * Todos os veículos monitorados.
 * Usado no painel público /veiculos e no dashboard.
 */
export function useVehicles() {
  return useQuery({
    queryKey: queryKeys.vehicles,
    queryFn: api.vehicles,
    staleTime: 5 * 60_000,   // 5 min — dados mudam a cada execução do pipeline
  });
}

/**
 * Um veículo específico.
 * Usado na página /veiculos/:ideology_id
 */
export function useVehicle(ideologyId: string) {
  return useQuery({
    queryKey: queryKeys.vehicle(ideologyId),
    queryFn: () => api.vehicle(ideologyId),
    enabled: !!ideologyId,
    staleTime: 5 * 60_000,
  });
}

/**
 * Espectro ideológico ordenado esquerda→direita.
 * Usado no componente SpectrumAxis.
 */
export function useSpectrum() {
  return useQuery({
    queryKey: queryKeys.spectrum,
    queryFn: api.spectrum,
    staleTime: 5 * 60_000,
  });
}

/**
 * Artigos recentes de um veículo.
 * @param ideologyId  ideology_id do veículo
 * @param limit       quantidade (padrão 20)
 * @param enabled     false para suspender a query (ex: usuário sem plano)
 */
export function useArticles(
  ideologyId: string,
  limit = 20,
  enabled = true,
) {
  return useQuery({
    queryKey: queryKeys.articles(ideologyId, limit),
    queryFn: () => api.articles(ideologyId, limit),
    enabled: enabled && !!ideologyId,
    staleTime: 2 * 60_000,  // artigos mudam com mais frequência
  });
}
