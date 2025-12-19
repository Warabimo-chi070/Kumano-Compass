// ==== Supabase 設定 ====
// ★ここをあなたの Supabase URL / KEY に変える
const SUPABASE_URL = "https://vxtrpopbdceqeeituudm.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ4dHJwb3BiZGNlcWVlaXR1dWRtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMyNTU0OTQsImV4cCI6MjA3ODgzMTQ5NH0.ofQ70uxgZsKueiBaUWQXGfyDcDsLlg3x_-s-UYaUZP0";

const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_KEY);


// ==== 最新20件をロード ====
async function loadVoices() {
  const { data, error } = await supabase
    .from("voices")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) {
    console.error("Supabase fetch error:", error);
    return;
  }

  renderTable(data);
}


// ==== テーブル描画 ====
function renderTable(rows) {
  const tbody = document.getElementById("voiceBody");
  tbody.innerHTML = "";

  rows.forEach(row => {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>${row.id}</td>
      <td>${row.text}</td>
      <td>${row.category ?? "-"}</td>
      <td>${row.sentiment ?? "-"}</td>
      <td>${row.importance_score ?? "-"}</td>
    `;

    tbody.appendChild(tr);
  });
}


// ==== 初回ロード ====
loadVoices();
