import { spawn } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const chromePath = process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const sourceUrl = process.env.REPLAY_URL ?? "http://127.0.0.1:4173/present/clear-before-commitment-v1?capture=1&static=1";
const outputPath = path.resolve(process.env.OUTPUT_PATH ?? "../docs/assets/beforecommit-full-replay-2x.mp4");
const frameDurationMs = 90;
const captureWidth = 1920;
const captureHeight = 1080;

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForDevTools(profileDirectory) {
  const activePort = path.join(profileDirectory, "DevToolsActivePort");
  for (let attempt = 0; attempt < 200; attempt += 1) {
    try {
      const [port] = (await readFile(activePort, "utf8")).trim().split("\n");
      return Number(port);
    } catch {
      await wait(50);
    }
  }
  throw new Error("Chrome did not expose a DevTools port.");
}

async function waitForTarget(port) {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json());
    const target = targets.find((candidate) => candidate.type === "page" && candidate.url.includes("/present/"));
    if (target) return target.webSocketDebuggerUrl;
    await wait(50);
  }
  throw new Error("The replay page did not appear in Chrome.");
}

function createCdpClient(url) {
  const socket = new WebSocket(url);
  let commandId = 0;
  const pending = new Map();
  const ready = new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(String(event.data));
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  return {
    ready,
    close: () => socket.close(),
    send(method, params = {}) {
      commandId += 1;
      return new Promise((resolve, reject) => {
        pending.set(commandId, { resolve, reject });
        socket.send(JSON.stringify({ id: commandId, method, params }));
      });
    },
  };
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", (code) => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)));
  });
}

const temporaryRoot = await mkdtemp(path.join(tmpdir(), "beforecommit-video-"));
const profileDirectory = path.join(temporaryRoot, "chrome-profile");
const frameDirectory = path.join(temporaryRoot, "frames");
await mkdir(profileDirectory, { recursive: true });
await mkdir(frameDirectory, { recursive: true });
await mkdir(path.dirname(outputPath), { recursive: true });

const chrome = spawn(chromePath, [
  "--headless=new",
  "--hide-scrollbars",
  "--disable-gpu",
  "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding",
  "--force-device-scale-factor=1",
  `--window-size=${captureWidth},${captureHeight}`,
  "--remote-debugging-port=0",
  `--user-data-dir=${profileDirectory}`,
  sourceUrl,
], { stdio: "ignore" });

let client;
try {
  const port = await waitForDevTools(profileDirectory);
  const websocketUrl = await waitForTarget(port);
  client = createCdpClient(websocketUrl);
  await client.ready;
  await client.send("Page.enable");
  await client.send("Runtime.enable");
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: captureWidth,
    height: captureHeight,
    deviceScaleFactor: 1,
    mobile: false,
  });
  const frameCount = await evaluate(client, `(async () => {
    const deadline = Date.now() + 30000;
    while ((!document.querySelector('.timeline-input') || !document.querySelector('.replay-canvas canvas')) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    await document.fonts.ready;
    const input = document.querySelector('.timeline-input');
    if (!input) throw new Error('Replay timeline did not load.');
    return Number(input.max) + 1;
  })()`);

  for (let frame = 0; frame < frameCount; frame += 1) {
    await evaluate(client, `(async () => {
      const input = document.querySelector('.timeline-input');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(input, ${frame});
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    })()`);
    const captured = await client.send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: false,
    });
    await writeFile(path.join(frameDirectory, `frame-${String(frame).padStart(4, "0")}.png`), Buffer.from(captured.data, "base64"));
    if (frame % 40 === 0 || frame === frameCount - 1) {
      process.stdout.write(`Captured ${frame + 1}/${frameCount} replay frames\n`);
    }
  }

  await run("ffmpeg", [
    "-y",
    "-framerate", `${1000 / frameDurationMs}`,
    "-start_number", "0",
    "-i", path.join(frameDirectory, "frame-%04d.png"),
    "-vf", "fps=60,format=yuv420p",
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "18",
    "-movflags", "+faststart",
    "-an",
    outputPath,
  ]);
  process.stdout.write(`Wrote ${outputPath}\n`);
} finally {
  client?.close();
  if (chrome.exitCode === null) {
    chrome.kill("SIGTERM");
    await Promise.race([
      new Promise((resolve) => chrome.once("exit", resolve)),
      wait(2000),
    ]);
  }
  for (let attempt = 0; attempt < 10; attempt += 1) {
    try {
      await rm(temporaryRoot, { recursive: true, force: true });
      break;
    } catch (error) {
      if (attempt === 9) throw error;
      await wait(100);
    }
  }
}
