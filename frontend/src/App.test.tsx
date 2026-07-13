import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

describe("App", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows service readiness and classifies a suspicious example", async () => {
    fetchMock
      .mockResolvedValueOnce(response({ status: "ready" }))
      .mockResolvedValueOnce(response({ label: "spam", confidence: 0.943 }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("Service ready")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Suspicious promotion" }));
    await user.click(screen.getByRole("button", { name: "Run classification" }));

    expect(await screen.findByRole("heading", { name: "Likely spam" })).toBeInTheDocument();
    expect(screen.getByText("94.3%")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Model confidence" })).toHaveAttribute(
      "aria-valuenow",
      "94",
    );
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/predict",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("Congratulations"),
      }),
    );
  });

  it("validates empty input without calling the prediction endpoint", async () => {
    fetchMock.mockResolvedValueOnce(response({ status: "ready" }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Run classification" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Enter an SMS message");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });

  it("explains how to recover when the model is unavailable", async () => {
    fetchMock
      .mockResolvedValueOnce(response({ status: "not_ready" }, 503))
      .mockResolvedValueOnce(response({ detail: "Prediction model is unavailable." }, 503));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("Service unavailable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Personal message" }));
    await user.click(screen.getByRole("button", { name: "Run classification" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Train or mount the model");
  });
});
