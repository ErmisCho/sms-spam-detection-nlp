export type PredictionLabel = "ham" | "spam";

export interface PredictionResponse {
  label: PredictionLabel;
  confidence: number;
}

export interface TimedPrediction extends PredictionResponse {
  latencyMs: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? "The service could not complete this request.";
  } catch {
    return "The service could not complete this request.";
  }
}

export async function checkReadiness(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/health/ready", { signal });
    return response.ok;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    return false;
  }
}

export async function predictSms(text: string): Promise<TimedPrediction> {
  const started = performance.now();
  const response = await fetch("/api/v1/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }

  const prediction = (await response.json()) as PredictionResponse;
  return {
    ...prediction,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
  };
}
