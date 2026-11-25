"""Interface web moderna e simplificada para instrutores e monitores."""

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Acesso Restrito — UberHub Calibrador</title>
  <style>
    :root {
      --bg: #f7f6f2;
      --surface: #ffffff;
      --border: #dcd8ce;
      --text: #14181f;
      --muted: #4b5563;
      --primary: #0f4c81;
      --primary-hover: #0b3a63;
      --danger: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.25rem;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .login-card {
      width: 100%;
      max-width: 400px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.75rem;
      box-shadow: 0 2px 8px rgba(20, 24, 31, 0.06);
    }
    .tag {
      display: inline-block;
      font-size: 0.8125rem;
      font-weight: 700;
      color: var(--primary);
      background: #e8f1f8;
      border: 1px solid #bfd7ec;
      padding: 0.18rem 0.6rem;
      border-radius: 999px;
      margin-bottom: 0.75rem;
    }
    h1 { font-size: 1.35rem; margin: 0 0 0.35rem; font-weight: 800; }
    p { margin: 0 0 1.25rem; color: var(--muted); font-size: 0.9375rem; }
    label { display: block; font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.35rem; }
    input[type="password"] {
      width: 100%;
      padding: 0.7rem 0.85rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 16px;
      margin-bottom: 0.9rem;
    }
    input[type="password"]:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(15, 76, 129, 0.12);
    }
    button {
      width: 100%;
      background: var(--primary);
      color: #ffffff;
      border: none;
      border-radius: 10px;
      padding: 0.78rem;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--primary-hover); }
    button:disabled { opacity: 0.6; cursor: wait; }
    .err {
      margin-top: 0.85rem;
      color: var(--danger);
      font-size: 0.875rem;
      font-weight: 600;
      display: none;
    }
  </style>
</head>
<body>
  <form class="login-card" onsubmit="doLogin(event)">
    <span class="tag">UberHub Code Club</span>
    <h1>Acesso para Instrutores</h1>
    <p>Digite a senha da equipe para acessar o Calibrador de Questões.</p>
    <label for="pwd">Senha de acesso</label>
    <input type="password" id="pwd" required autocomplete="current-password" placeholder="••••••••" />
    <button type="submit" id="login-btn">Entrar</button>
    <div class="err" id="login-err"></div>
  </form>
  <script>
    async function doLogin(e) {
      e.preventDefault();
      const btn = document.getElementById('login-btn');
      const err = document.getElementById('login-err');
      err.style.display = 'none';
      btn.disabled = true;
      try {
        const resp = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: document.getElementById('pwd').value }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Senha inválida');
        window.location.reload();
      } catch (ex) {
        err.textContent = ex.message;
        err.style.display = 'block';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Calibrador de Questões — Listas & Maratonas</title>
  <style>
    :root {
      --bg: #f7f6f2;
      --surface: #ffffff;
      --surface-alt: #efede7;
      --border: #dcd8ce;
      --text: #14181f;
      --muted: #4b5563;
      --primary: #0f4c81;
      --primary-hover: #0b3a63;
      --primary-soft: #e8f1f8;
      --success: #15803d;
      --danger: #b91c1c;
      --shadow: 0 1px 2px rgba(20, 24, 31, 0.05), 0 6px 16px rgba(20, 24, 31, 0.04);
      --radius: 12px;
      --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      overflow-x: hidden;
      font-family: var(--font-sans);
      font-size: 15px;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    code { font-family: var(--font-mono); font-size: 0.85em; }
    .page { max-width: 1120px; margin: 0 auto; padding: 1.25rem 1.25rem 3rem; }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      padding-bottom: 1rem;
      margin-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: center; gap: 0.6rem; min-width: 0; }
    .wordmark {
      font-size: 1.375rem;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.025em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .pill-tag {
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 0.8125rem;
      font-weight: 700;
      padding: 0.18rem 0.6rem;
      border-radius: 999px;
      border: 1px solid #bfd7ec;
      white-space: nowrap;
    }
    .toggle-wrap {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--text);
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 0.4rem 0.75rem;
      border-radius: 999px;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .toggle-wrap input { accent-color: var(--primary); cursor: pointer; }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 1.25rem;
      align-items: start;
    }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.35rem;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .card-title {
      font-size: 1rem;
      font-weight: 700;
      margin: 0 0 0.9rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .segmented {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      background: var(--surface-alt);
      padding: 0.25rem;
      border-radius: 10px;
      margin-bottom: 1rem;
      gap: 0.25rem;
    }
    @media (max-width: 680px) {
      .page { padding: 0.9rem 0.85rem 2.25rem; }
      .card { padding: 1rem; }
      .wordmark { font-size: 1.15rem; }
      .pill-tag { display: none; }
      .segmented { grid-template-columns: 1fr 1fr; }
      .row { grid-template-columns: 1fr; gap: 0; }
      textarea, input[type="text"], input[type="number"] { font-size: 16px; }
    }
    .seg-btn {
      border: none;
      background: transparent;
      color: var(--muted);
      font-weight: 600;
      font-size: 0.88rem;
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .seg-btn.active {
      background: var(--surface);
      color: var(--text);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }

    .examples-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 1rem;
    }
    .examples-label { font-size: 0.78rem; color: var(--muted); font-weight: 600; margin-right: 0.2rem; }
    .chip-btn {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.25rem 0.65rem;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.12s ease;
    }
    .chip-btn:hover { border-color: var(--primary); color: var(--primary); background: var(--primary-soft); }

    label { display: block; font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem; }
    .hint { font-weight: 400; color: var(--muted); }
    textarea, input[type="text"], input[type="number"] {
      width: 100%;
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.65rem 0.8rem;
      font-size: 0.92rem;
      font-family: inherit;
      margin-bottom: 0.85rem;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    textarea:focus, input:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    .row { display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 0.85rem; }
    .presets { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: -0.45rem 0 0.9rem; }

    .dropzone {
      border: 2px dashed #cbd5e1;
      background: var(--surface-alt);
      border-radius: 12px;
      padding: 1.75rem 1rem;
      text-align: center;
      cursor: pointer;
      margin-bottom: 0.9rem;
      color: var(--muted);
      transition: all 0.15s ease;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: var(--primary-soft);
      color: var(--primary);
    }
    #preview-img { max-width: 100%; max-height: 210px; margin-top: 0.75rem; border-radius: 8px; border: 1px solid var(--border); }

    .btn-primary {
      background: var(--primary);
      color: #ffffff;
      border: none;
      padding: 0.78rem 1.2rem;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      width: 100%;
      transition: background 0.15s ease;
    }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-primary:disabled { opacity: 0.6; cursor: wait; }

    .btn-secondary {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 0.85rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.84rem;
      cursor: pointer;
    }
    .btn-secondary:hover { border-color: var(--primary); color: var(--primary); }

    .hidden { display: none !important; }

    .empty-state {
      text-align: center;
      padding: 2.2rem 1rem;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }

    .result-box {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.15rem;
      background: var(--surface);
      margin-bottom: 0.85rem;
    }
    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
    }
    .level-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.35rem 0.8rem;
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.95rem;
    }
    .rating-pill {
      font-size: 1.1rem;
      font-weight: 800;
      color: var(--text);
      background: var(--surface-alt);
      padding: 0.3rem 0.7rem;
      border-radius: 8px;
      border: 1px solid var(--border);
    }

    .meter-wrap { margin: 0.65rem 0 0.85rem; }
    .meter-track {
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(90deg, #94a3b8 0%, #22c55e 25%, #0ea5e9 45%, #8b5cf6 70%, #ef4444 100%);
      position: relative;
    }
    .meter-thumb {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #ffffff;
      border: 3px solid var(--text);
      position: absolute;
      top: 50%;
      transform: translate(-50%, -50%);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    }
    .meter-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--muted);
      margin-top: 0.3rem;
    }

    .track-banner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.4rem;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.45rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.75rem;
    }
    .similar-box {
      margin-top: 0.8rem;
      padding: 0.7rem 0.85rem;
      background: var(--surface-alt);
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .similar-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 0.45rem;
    }
    .similar-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-size: 0.82rem;
      padding: 0.3rem 0;
      border-bottom: 1px solid #e2e8f0;
    }
    .similar-row:last-child { border-bottom: none; }
    .alert-box {
      background: #fffbeb;
      border: 1px solid #fde68a;
      color: #92400e;
      border-radius: 10px;
      padding: 0.65rem 0.85rem;
      font-size: 0.84rem;
      margin-bottom: 0.75rem;
    }
    .spinner {
      width: 34px;
      height: 34px;
      border: 3px solid var(--border);
      border-top-color: var(--primary);
      border-radius: 50%;
      margin: 0 auto 0.75rem;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .coach-box {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 10px;
      padding: 0.75rem 0.9rem;
      margin-bottom: 0.85rem;
      font-size: 0.86rem;
      color: #14532d;
    }
    .coach-box strong { color: #166534; }
    .guidance {
      background: var(--primary-soft);
      border-left: 3px solid var(--primary);
      padding: 0.65rem 0.85rem;
      border-radius: 0 8px 8px 0;
      font-size: 0.88rem;
      color: #1e3a8a;
      margin-bottom: 0.85rem;
    }
    .signals { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.85rem; }
    .signal-chip {
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.22rem 0.58rem;
      border-radius: 6px;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .result-actions { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    .feedback-bar {
      margin-top: 0.75rem;
      padding-top: 0.65rem;
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0.45rem;
      flex-wrap: wrap;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .fb-btn {
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.2rem 0.6rem;
      font-size: 0.78rem;
      cursor: pointer;
      color: var(--text);
    }
    .fb-btn:hover { border-color: var(--primary); color: var(--primary); }

    .tech-details {
      margin-top: 0.8rem;
      padding-top: 0.7rem;
      border-top: 1px dashed var(--border);
      font-size: 0.82rem;
      color: var(--muted);
    }
    .ocr-edit-area { margin-top: 0.75rem; }
    .ocr-edit-area textarea { font-size: 0.84rem; margin-bottom: 0.5rem; }

    .list-card { margin-top: 1.25rem; }
    .list-summary { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.75rem; }
    .list-items { display: flex; flex-direction: column; gap: 0.5rem; }
    .list-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.6rem;
      padding: 0.65rem 0.85rem;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 0.88rem;
    }
    .list-row-left { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
    .letter-badge {
      width: 26px;
      height: 26px;
      border-radius: 7px;
      background: var(--text);
      color: #ffffff;
      font-weight: 800;
      font-size: 0.82rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .remove-btn {
      background: transparent;
      border: none;
      color: var(--muted);
      font-size: 1.1rem;
      cursor: pointer;
      padding: 0.1rem 0.4rem;
    }
    .remove-btn:hover { color: var(--danger); }
    .error-msg {
      background: #fef2f2;
      color: var(--danger);
      border: 1px solid #fecaca;
      border-radius: 10px;
      padding: 0.75rem 0.9rem;
      font-size: 0.88rem;
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <h1 class="wordmark">Calibrador de Questões</h1>
        <span class="pill-tag">UberHub Code Club</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.5rem;flex-shrink:0">
        <label class="toggle-wrap">
          <input type="checkbox" id="tech-toggle" onchange="toggleTechMode()" />
          <span>Modo técnico</span>
        </label>
        <button type="button" class="btn-secondary __AUTH_LOGOUT_CLASS__" onclick="doLogout()">Sair</button>
      </div>
    </header>

    <div class="grid">
      <!-- Coluna Esquerda: Entrada -->
      <div class="card">
        <div class="segmented">
          <button class="seg-btn active" id="seg-text" onclick="switchTab('text')">Texto</button>
          <button class="seg-btn" id="seg-url" onclick="switchTab('url')">Link</button>
          <button class="seg-btn" id="seg-batch" onclick="switchTab('batch')">Lista (Lote)</button>
          <button class="seg-btn" id="seg-image" onclick="switchTab('image')">Print</button>
        </div>

        <div id="tab-text">
          <div class="examples-bar">
            <span class="examples-label">Exemplos rápidos:</span>
            <button class="chip-btn" onclick="loadExample('easy')">Iniciante</button>
            <button class="chip-btn" onclick="loadExample('bs')">Busca Binária</button>
            <button class="chip-btn" onclick="loadExample('graph')">Grafos / Dijkstra</button>
            <button class="chip-btn" onclick="loadExample('tree')">Árvore + DP</button>
          </div>

          <label for="prob-title">Nome da questão <span class="hint">(opcional, para sua lista)</span></label>
          <input type="text" id="prob-title" placeholder="Ex: Questão C - Caminho das Cidades" />

          <label for="statement">Enunciado ou resumo do problema</label>
          <textarea id="statement" rows="5"
            placeholder="Cole o enunciado aqui (em português ou inglês). Ex: Dada uma árvore com n vértices, encontre o diâmetro e responda q consultas."
          ></textarea>

          <div class="row">
            <div>
              <label for="n-limit">Maior limite de N <span class="hint">(opcional)</span></label>
              <input type="text" id="n-limit" placeholder="Ex: 10^5, 20, 10^18" />
            </div>
            <div>
              <label for="time-limit">Tempo limite (s)</label>
              <input type="number" id="time-limit" value="2.0" step="0.5" min="0.5" max="30" />
            </div>
          </div>

          <div class="presets">
            <button type="button" class="chip-btn" onclick="setNLimit('20')">N ≤ 20</button>
            <button type="button" class="chip-btn" onclick="setNLimit('1000')">N ≤ 10³</button>
            <button type="button" class="chip-btn" onclick="setNLimit('10^5')">N ≤ 10⁵</button>
            <button type="button" class="chip-btn" onclick="setNLimit('2 * 10^5')">N ≤ 2·10⁵</button>
            <button type="button" class="chip-btn" onclick="setNLimit('10^{18}')">N ≤ 10¹⁸</button>
          </div>

          <button class="btn-primary" id="btn-text" onclick="predictText()">Estimar Dificuldade</button>
        </div>

        <div id="tab-url" class="hidden">
          <div class="examples-bar">
            <span class="examples-label">Testar link:</span>
            <button class="chip-btn" onclick="loadUrlExample('https://codeforces.com/problemset/problem/4/A')">CF 4A</button>
            <button class="chip-btn" onclick="loadUrlExample('https://codeforces.com/problemset/problem/1850/E')">CF 1850E</button>
            <button class="chip-btn" onclick="loadUrlExample('https://atcoder.jp/contests/abc300/tasks/abc300_c')">AtCoder ABC300 C</button>
          </div>
          <label for="prob-url">Link da questão <span class="hint">(Codeforces, AtCoder, NEPS, Beecrowd)</span></label>
          <input type="text" id="prob-url" placeholder="https://codeforces.com/problemset/problem/158/A" />
          <p style="font-size:0.82rem;color:var(--muted);margin:-0.35rem 0 0.9rem">
            Para <strong>Codeforces</strong> e <strong>AtCoder</strong>, consultamos o rating oficial direto da API. Para outros sites, lemos o enunciado automaticamente quando público.
          </p>
          <button class="btn-primary" id="btn-url" onclick="predictUrl()">Consultar Link</button>
        </div>

        <div id="tab-batch" class="hidden">
          <div class="examples-bar">
            <span class="examples-label">Exemplo de lista:</span>
            <button class="chip-btn" onclick="loadBatchExample()">Preencher 4 questões</button>
          </div>
          <label for="batch-input">Cole links (CF/AtCoder/NEPS) ou resumos <span class="hint">(1 questão por linha, até 15)</span></label>
          <textarea id="batch-input" rows="6"
            placeholder="https://codeforces.com/problemset/problem/4/A&#10;https://codeforces.com/problemset/problem/1850/E&#10;https://atcoder.jp/contests/abc300/tasks/abc300_c&#10;Um carteiro precisa ir da cidade 1 até N com menor pedágio. 1 <= N <= 10^5."
          ></textarea>
          <button class="btn-primary" id="btn-batch" onclick="analyzeBatch()">Analisar e Ordenar Lista Completa</button>
        </div>

        <div id="tab-image" class="hidden">
          <div class="dropzone" id="dropzone" onclick="document.getElementById('img-input').click()">
            <div><strong>Clique para enviar</strong>, arraste um arquivo ou pressione <strong>Ctrl+V</strong> com um print</div>
            <div style="font-size:0.8rem;margin-top:0.25rem">Suporta PNG, JPG, WEBP (processamento 100% local)</div>
            <input type="file" id="img-input" accept="image/*" class="hidden" onchange="previewFile(event)" />
            <div><img id="preview-img" class="hidden" alt="Preview da questão" /></div>
          </div>

          <label for="img-n-limit">Forçar limite de N <span class="hint">(opcional, se o print cortar ou errar o expoente)</span></label>
          <input type="text" id="img-n-limit" placeholder="Ex: 10^5" />
          <button class="btn-primary" id="btn-image" onclick="predictImage()">Ler Print e Estimar</button>
        </div>
      </div>

      <!-- Coluna Direita: Resultado -->
      <div class="card">
        <div class="card-title">
          <span>Resultado da Análise</span>
        </div>
        <div id="results">
          <div class="empty-state">
            <div class="empty-icon">🎯</div>
            <strong>Pronto para calibrar</strong>
            <p style="margin:0.35rem 0 0">
              Cole o enunciado, escolha um exemplo rápido ou dê <strong>Ctrl+V</strong> em um print de questão para ver o nível recomendado.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Lista da Aula / Prova -->
    <div class="card list-card hidden" id="contest-card">
      <div class="card-title">
        <span>📋 Lista da Aula / Prova (ordenada por dificuldade)</span>
        <div style="display:flex;gap:0.5rem">
          <button class="btn-secondary" id="copy-list-btn" onclick="copyContestList()">Copiar Resumo</button>
          <button class="btn-secondary" onclick="clearContestList()">Limpar</button>
        </div>
      </div>
      <div class="list-summary" id="contest-summary"></div>
      <div id="contest-alerts"></div>
      <div class="list-items" id="contest-items"></div>
    </div>
  </div>

  <script>
    const EXAMPLES = {
      easy: {
        title: 'Soma e Contagem de Pares',
        statement: 'Leia um vetor de n números inteiros e imprima quantos elementos são pares e a soma total.',
        n_limit: '1000',
        time_limit: 1.0,
      },
      bs: {
        title: 'Divisão de Blocos',
        statement: 'Encontre o menor valor x para dividir o vetor em k blocos usando busca binária na resposta monótona.',
        n_limit: '2 * 10^5',
        time_limit: 2.0,
      },
      graph: {
        title: 'Menor Caminho entre Cidades',
        statement: 'Dado um grafo conexo com n vértices e m estradas com pesos, encontre o menor caminho entre a cidade 1 e n usando Dijkstra.',
        n_limit: '10^5',
        time_limit: 2.0,
      },
      tree: {
        title: 'Diâmetro e DP em Árvore',
        statement: 'Dada uma árvore enraizada com n vértices, calcule o conjunto independente máximo e o menor ancestral comum (LCA) usando programação dinâmica.',
        n_limit: '10^5',
        time_limit: 2.0,
      },
    };

    let currentTab = 'text';
    let pastedFile = null;
    let lastItems = [];
    let contestList = [];

    function switchTab(tab) {
      currentTab = tab;
      ['text', 'url', 'batch', 'image'].forEach(t => {
        document.getElementById(`seg-${t}`).classList.toggle('active', tab === t);
        document.getElementById(`tab-${t}`).classList.toggle('hidden', tab !== t);
      });
    }

    function loadBatchExample() {
      switchTab('batch');
      document.getElementById('batch-input').value = [
        'https://codeforces.com/problemset/problem/4/A',
        'https://codeforces.com/problemset/problem/1850/E',
        'https://atcoder.jp/contests/abc300/tasks/abc300_c',
        'Um carteiro precisa entregar cartas em N cidades ligadas por M estradas bidirecionais com pedágios. Encontre o menor custo da cidade 1 até N. 1 <= N <= 10^5.'
      ].join('\\n');
    }

    function loadUrlExample(url) {
      switchTab('url');
      document.getElementById('prob-url').value = url;
      predictUrl();
    }

    function setNLimit(val) {
      document.getElementById('n-limit').value = val;
    }

    function loadExample(key) {
      const ex = EXAMPLES[key];
      if (!ex) return;
      switchTab('text');
      document.getElementById('prob-title').value = ex.title;
      document.getElementById('statement').value = ex.statement;
      document.getElementById('n-limit').value = ex.n_limit;
      document.getElementById('time-limit').value = ex.time_limit;
      predictText();
    }

    function toggleTechMode() {
      const show = document.getElementById('tech-toggle').checked;
      document.querySelectorAll('.tech-details').forEach(el => el.classList.toggle('hidden', !show));
    }

    async function doLogout() {
      await fetch('/api/logout', { method: 'POST' });
      window.location.reload();
    }

    function handleImageFile(file) {
      if (!file || !file.type.startsWith('image/')) return;
      pastedFile = file;
      const img = document.getElementById('preview-img');
      img.src = URL.createObjectURL(file);
      img.classList.remove('hidden');
      if (currentTab !== 'image') switchTab('image');
    }

    function previewFile(event) {
      handleImageFile(event.target.files[0]);
    }

    // Suporte a Ctrl+V de prints em qualquer lugar da pagina
    document.addEventListener('paste', (event) => {
      const items = (event.clipboardData || {}).items || [];
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            handleImageFile(file);
            predictImage();
            break;
          }
        }
      }
    });

    const dz = document.getElementById('dropzone');
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('dragover');
      if (e.dataTransfer.files[0]) handleImageFile(e.dataTransfer.files[0]);
    });

    function levelStyle(rating) {
      if (rating < 1200) return { bg: '#f1f5f9', fg: '#334155', border: '#cbd5e1' };
      if (rating < 1600) return { bg: '#dcfce7', fg: '#166534', border: '#86efac' };
      if (rating < 2000) return { bg: '#dbeafe', fg: '#1e40af', border: '#93c5fd' };
      if (rating < 2400) return { bg: '#f3e8ff', fg: '#6b21a8', border: '#d8b4fe' };
      return { bg: '#fee2e2', fg: '#991b1b', border: '#fca5a5' };
    }

    function escapeHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function renderItems(items) {
      lastItems = items;
      const showTech = document.getElementById('tech-toggle').checked;
      const container = document.getElementById('results');

      container.innerHTML = items.map((item, idx) => {
        const st = levelStyle(item.rating);
        const pct = Math.max(0, Math.min(100, ((item.rating - 800) / 2000) * 100));
        const topicsHtml = (item.topics || []).map(t => `<span class="signal-chip">🏷️ ${escapeHtml(t)}</span>`).join('');
        const titleLabel = item.title || item.label || '';

        return `
          <div class="result-box">
            <div class="result-header">
              <span class="level-badge" style="background:${st.bg};color:${st.fg};border:1px solid ${st.border}">
                ${titleLabel ? escapeHtml(titleLabel) + ' • ' : ''}${escapeHtml(item.level)}
              </span>
              <span class="rating-pill">~${item.rating}</span>
            </div>

            <div class="meter-wrap">
              <div class="meter-track"><div class="meter-thumb" style="left:${pct}%"></div></div>
              <div class="meter-labels">
                <span>Iniciante (800)</span>
                <span>Intermediário (1400)</span>
                <span>Avançado (1900)</span>
                <span>ICPC (2400+)</span>
              </div>
            </div>

            ${item.track ? `
              <div class="track-banner">
                <span>🎓 ${escapeHtml(item.track)}</span>
                <span style="color:var(--primary)">${escapeHtml(item.slot || '')}</span>
              </div>
            ` : ''}

            ${item.guidance ? `<div class="guidance">${escapeHtml(item.guidance)}</div>` : ''}

            ${(item.algorithm || item.complexity || item.mentor_tip) ? `
              <div class="coach-box">
                ${item.algorithm ? `<div><strong>🧠 Técnica esperada:</strong> ${escapeHtml(item.algorithm)} ${item.complexity ? `<code>(${escapeHtml(item.complexity)})</code>` : ''}</div>` : ''}
                ${item.mentor_tip ? `<div style="margin-top:0.3rem"><strong>💡 Dica para monitoria:</strong> ${escapeHtml(item.mentor_tip)}</div>` : ''}
              </div>
            ` : ''}

            <div class="signals">
              ${topicsHtml}
              <span class="signal-chip">📏 ${escapeHtml(item.n_display || 'N não informado')}</span>
              ${item.time_limit ? `<span class="signal-chip">⏱️ ${item.time_limit}s</span>` : ''}
              ${item.confidence ? `<span class="signal-chip">🎯 ${escapeHtml(item.confidence)}</span>` : ''}
            </div>

            ${(item.similar_problems && item.similar_problems.length) ? `
              <div class="similar-box">
                <div class="similar-title">Questões reais parecidas (Referência Codeforces)</div>
                ${item.similar_problems.map(sp => `
                  <div class="similar-row">
                    <span><strong>${escapeHtml(sp.title)}</strong> <span style="color:var(--muted)">(${escapeHtml((sp.tags || []).join(', '))})</span></span>
                    <span style="white-space:nowrap"><b>${sp.rating}</b> · ${sp.similarity}%</span>
                  </div>
                `).join('')}
              </div>
            ` : ''}

            <div class="result-actions">
              <button type="button" class="btn-secondary" onclick="addToContest(${idx})">+ Adicionar à Lista da Aula</button>
              ${item.ocr_text ? `<button type="button" class="btn-secondary" onclick="toggleOcrEdit(${idx})">Editar texto do OCR</button>` : ''}
            </div>

            <div class="feedback-bar" id="fb-bar-${idx}">
              <span>O nível parece certo?</span>
              <button type="button" class="fb-btn" onclick="sendFeedback(${idx}, 'ok')">👍 Certo</button>
              <button type="button" class="fb-btn" onclick="sendFeedback(${idx}, 'too_high')">⬇️ Muito alto</button>
              <button type="button" class="fb-btn" onclick="sendFeedback(${idx}, 'too_low')">⬆️ Muito baixo</button>
            </div>

            ${item.ocr_text ? `
              <div class="ocr-edit-area hidden" id="ocr-edit-${idx}">
                <label>Texto extraído (corrija se necessário e recalcule):</label>
                <textarea id="ocr-textarea-${idx}" rows="4">${escapeHtml(item.ocr_text)}</textarea>
                <button type="button" class="btn-secondary" onclick="recalcFromOcr(${idx})">Recalcular com texto editado</button>
              </div>
            ` : ''}

            <div class="tech-details ${showTech ? '' : 'hidden'}">
              ${item.ai_engine ? `Motor: <b>${escapeHtml(item.ai_engine)}</b> • ` : ''}
              ${item.llm_rating ? `LLM Local: <b>${item.llm_rating}</b> • ` : ''}
              XGBoost: <b>${item.xgb_rating}</b> • PyTorch MLP: <b>${item.mlp_rating}</b> • log10(N): <b>${item.max_n_log10}</b>
            </div>
          </div>
        `;
      }).join('');
    }

    function toggleOcrEdit(idx) {
      document.getElementById(`ocr-edit-${idx}`).classList.toggle('hidden');
    }

    async function recalcFromOcr(idx) {
      const text = document.getElementById(`ocr-textarea-${idx}`).value.trim();
      if (!text) return;
      const item = lastItems[idx];
      const nLim = document.getElementById('img-n-limit').value.trim() || null;
      document.getElementById('results').innerHTML = `
        <div class="empty-state">
          <div class="spinner"></div>
          <strong>Recalculando com texto corrigido...</strong>
        </div>`;
      try {
        const resp = await fetch('/api/predict-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ statement: text, n_limit: nLim, time_limit: item.time_limit || 2.0 }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro ao recalcular');
        lastItems[idx] = { ...data, label: item.label, title: item.title, time_limit: item.time_limit, ocr_text: text };
        renderItems(lastItems);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
      }
    }

    function addToContest(idx) {
      const item = lastItems[idx];
      if (!item) return;
      const fallbackTitle = `Questão ${contestList.length + 1}`;
      contestList.push({
        title: item.title || item.label || fallbackTitle,
        rating: item.rating,
        level: item.level,
        topics: item.topics || [],
        n_display: item.n_display || '',
      });
      contestList.sort((a, b) => a.rating - b.rating);
      renderContestList();
    }

    function removeFromContest(idx) {
      contestList.splice(idx, 1);
      renderContestList();
    }

    function clearContestList() {
      contestList = [];
      renderContestList();
    }

    function renderContestList(alerts = []) {
      const card = document.getElementById('contest-card');
      if (contestList.length === 0) { card.classList.add('hidden'); return; }
      card.classList.remove('hidden');

      const counts = {};
      contestList.forEach(q => { counts[q.level] = (counts[q.level] || 0) + 1; });
      const breakdown = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(' • ');
      document.getElementById('contest-summary').textContent =
        `${contestList.length} ${contestList.length === 1 ? 'questão' : 'questões'} na lista: ${breakdown}`;

      const alertsEl = document.getElementById('contest-alerts');
      alertsEl.innerHTML = (alerts || []).map(a => `<div class="alert-box">⚠️ ${escapeHtml(a)}</div>`).join('');

      document.getElementById('contest-items').innerHTML = contestList.map((q, i) => {
        const letter = String.fromCharCode(65 + (i % 26));
        const st = levelStyle(q.rating);
        return `
          <div class="list-row">
            <div class="list-row-left">
              <span class="letter-badge">${letter}</span>
              <strong>${escapeHtml(q.title)}</strong>
              <span class="signal-chip" style="background:${st.bg};color:${st.fg};border-color:${st.border}">
                ${escapeHtml(q.level)} (~${q.rating})
              </span>
              <span style="color:var(--muted)">${escapeHtml((q.topics || []).slice(0, 2).join(', '))}</span>
            </div>
            <button class="remove-btn" title="Remover" onclick="removeFromContest(${i})">×</button>
          </div>
        `;
      }).join('');
    }

    async function copyContestList() {
      if (contestList.length === 0) return;
      const lines = ['*Lista Calibrada (ordenada por dificuldade):*'];
      contestList.forEach((q, i) => {
        const letter = String.fromCharCode(65 + (i % 26));
        const topics = (q.topics || []).slice(0, 2).join(', ');
        lines.push(`${letter}. ${q.title} — ${q.level} (~${q.rating}) [${topics}]`);
      });
      const text = lines.join('\\n');
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      const btn = document.getElementById('copy-list-btn');
      const prev = btn.textContent;
      btn.textContent = 'Copiado!';
      setTimeout(() => { btn.textContent = prev; }, 1500);
    }

    function computeContestAlerts(items) {
      const alerts = [];
      for (let i = 0; i < items.length - 1; i++) {
        const r1 = items[i].rating;
        const r2 = items[i + 1].rating;
        if (r2 - r1 >= 400) {
          const l1 = String.fromCharCode(65 + (i % 26));
          const l2 = String.fromCharCode(65 + ((i + 1) % 26));
          const mid = Math.round(((r1 + r2) / 2) / 50) * 50;
          alerts.push(`Salto brusco entre ${l1} (${r1}) e ${l2} (${r2}): considere incluir uma questão intermediária (~${mid}).`);
        }
      }
      if (items.length >= 3 && items[0].rating >= 1400) {
        alerts.push('A lista não tem questão de aquecimento (800–1100). Considere adicionar uma questão A mais acessível.');
      }
      return alerts;
    }

    async function analyzeBatch() {
      const btn = document.getElementById('btn-batch');
      const raw = document.getElementById('batch-input').value;
      const lines = raw.split(/\\r?\\n/).map(s => s.trim()).filter(Boolean).slice(0, 15);
      if (!lines.length) {
        document.getElementById('results').innerHTML = '<div class="error-msg">Cole ao menos 1 link ou enunciado (1 por linha).</div>';
        return;
      }
      btn.disabled = true;
      const items = [];
      const errors = [];
      try {
        for (let i = 0; i < lines.length; i++) {
          const entry = lines[i];
          document.getElementById('results').innerHTML = `
            <div class="empty-state">
              <div class="spinner"></div>
              <strong>Analisando questão ${i + 1} de ${lines.length}...</strong>
              <p style="margin:0.35rem 0 0">Pode deixar carregando — cada questão é processada sem risco de timeout.</p>
            </div>`;
          try {
            const isUrl = /^https?:\\/\\//i.test(entry);
            const resp = await fetch(isUrl ? '/api/predict-url' : '/api/predict-text', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(isUrl ? { url: entry } : { statement: entry }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Erro no item');
            const fallbackTitle = entry.length > 65 ? entry.slice(0, 65) + '...' : entry;
            items.push({ ...data, title: data.title || fallbackTitle });
          } catch (itemErr) {
            errors.push(`Item ${i + 1}: ${itemErr.message}`);
          }
        }
        if (!items.length) {
          throw new Error('Nenhuma questão válida foi analisada. ' + errors.slice(0, 2).join(' | '));
        }
        items.sort((a, b) => a.rating - b.rating);
        contestList = items.map(it => ({
          title: it.title || 'Questão',
          rating: it.rating,
          level: it.level,
          topics: it.topics || [],
          n_display: it.n_display || '',
        }));
        const alerts = computeContestAlerts(items);
        if (errors.length) {
          alerts.push(`${errors.length} item(ns) não puderam ser lidos: ${errors[0]}`);
        }
        renderContestList(alerts);
        renderItems(items.slice(0, 3));
        document.getElementById('contest-card').scrollIntoView({ behavior: 'smooth' });
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    async function predictUrl() {
      const btn = document.getElementById('btn-url');
      const url = document.getElementById('prob-url').value.trim();
      if (!url) {
        document.getElementById('results').innerHTML = '<div class="error-msg">Cole o link da questão antes de consultar.</div>';
        return;
      }
      btn.disabled = true;
      document.getElementById('results').innerHTML = '<div class="empty-state">Consultando questão...</div>';
      try {
        const resp = await fetch('/api/predict-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro ao consultar link');
        renderItems([data]);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    async function sendFeedback(idx, verdict) {
      const item = lastItems[idx];
      if (!item) return;
      const bar = document.getElementById(`fb-bar-${idx}`);
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rating: item.rating,
            verdict,
            title: item.title || item.label || '',
            statement_preview: (item.ocr_text || document.getElementById('statement').value || '').slice(0, 280),
          }),
        });
        bar.innerHTML = '<span style="color:var(--success);font-weight:600">✓ Obrigado! Feedback registrado para calibração.</span>';
      } catch (e) {
        bar.innerHTML = '<span>Não foi possível salvar o feedback agora.</span>';
      }
    }

    async function predictText() {
      const btn = document.getElementById('btn-text');
      const statement = document.getElementById('statement').value.trim();
      if (!statement) {
        document.getElementById('results').innerHTML = '<div class="error-msg">Cole ou digite o enunciado do problema antes de estimar.</div>';
        return;
      }
      btn.disabled = true;
      document.getElementById('results').innerHTML = `
        <div class="empty-state">
          <div class="spinner"></div>
          <strong>Analisando com Coach IA Local...</strong>
          <p style="margin:0.35rem 0 0">Lendo restrições e raciocinando sobre a complexidade (~10–20s).</p>
        </div>`;
      try {
        const timeLimit = parseFloat(document.getElementById('time-limit').value) || 2.0;
        const resp = await fetch('/api/predict-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            statement,
            n_limit: document.getElementById('n-limit').value.trim() || null,
            time_limit: timeLimit,
          }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro na previsão');
        const title = document.getElementById('prob-title').value.trim();
        renderItems([{ ...data, title, time_limit: timeLimit }]);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    async function predictImage() {
      const file = pastedFile || document.getElementById('img-input').files[0];
      if (!file) {
        document.getElementById('results').innerHTML = '<div class="error-msg">Selecione uma imagem ou pressione Ctrl+V para colar um print.</div>';
        return;
      }
      const btn = document.getElementById('btn-image');
      btn.disabled = true;
      document.getElementById('results').innerHTML = `
        <div class="empty-state">
          <div class="spinner"></div>
          <strong>Lendo print e analisando com IA Local...</strong>
          <p style="margin:0.35rem 0 0">Extraindo texto offline (OCR) e calibrando dificuldade.</p>
        </div>`;
      try {
        const form = new FormData();
        form.append('file', file, file.name || 'print.png');
        const nLim = document.getElementById('img-n-limit').value.trim();
        if (nLim) form.append('n_limit', nLim);
        const resp = await fetch('/api/predict-image', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro ao processar imagem');
        renderItems(data.results);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""
