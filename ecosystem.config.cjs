module.exports = {
  apps: [
    {
      name: "qwen3-asr",
      cwd: __dirname,
      script: "./.venv/bin/python",
      args: "-m qwen3_asr serve -d qwen3-asr-0.6b --host 0.0.0.0 --port 8000",
      interpreter: "none",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      watch: false,
      min_uptime: "15s",
      max_restarts: 10,
      restart_delay: 3000,
      kill_timeout: 15000,
      time: true,
      merge_logs: true,
      out_file: "./logs/qwen3-asr-out.log",
      error_file: "./logs/qwen3-asr-error.log",
      env: {
        PYTHONUNBUFFERED: "1",
        QWEN3_ASR_HOST: "0.0.0.0",
        QWEN3_ASR_PORT: "8000",
        QWEN3_ASR_MAX_UPLOAD_MB: "100",
      },
    },
  ],
};
