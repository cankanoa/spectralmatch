# LLM Prompt

Use this text to prompt LLM models with context about this codebase which includes function headers and docs.

<div style="margin-bottom: 1em; position: relative;">

  <div style="display: flex; align-items: center; gap: 1em; margin-bottom: 0.5em;">
    <button onclick="copyToClipboard()" style="border: 1px solid #ccc; padding: 0.5em 1em; border-radius: 4px; background: #fff; cursor: pointer;">📋 Copy</button>
    <p id="copy-success" style="color: green; display: none; margin: 0;">✅ Copied!</p>
  </div>

  <div style="margin-top: 1em;">
    <a href="/spectralmatch">← To Readme</a>
  </div>
  <pre id="copy-target" style="max-height: 300px; overflow: auto; background: #f5f5f5; padding: 1em; border-radius: 6px; border: 1px solid #ccc;">
{prompt_content}
  </pre>

</div>

<script>
function copyToClipboard() {
    const text = document.getElementById("copy-target").innerText;
    navigator.clipboard.writeText(text).then(function () {
        const successMsg = document.getElementById("copy-success");
        successMsg.style.display = "inline";
        setTimeout(() => successMsg.style.display = "none", 2000);
    });
}
</script>