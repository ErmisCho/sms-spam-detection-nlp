import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const environment = { ...process.env };

// Shared Windows/WSL shells can export a Windows temp alias that Linux cannot resolve.
if (process.platform !== "win32") {
  environment.TMPDIR = "/tmp";
  environment.TEMP = "/tmp";
  environment.TMP = "/tmp";
}

const vitest = fileURLToPath(new URL("../node_modules/vitest/vitest.mjs", import.meta.url));
const result = spawnSync(process.execPath, [vitest, "run"], {
  env: environment,
  stdio: "inherit",
});

process.exit(result.status ?? 1);
