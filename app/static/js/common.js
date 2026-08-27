/* 全站公共逻辑：运行设备提示与关闭程序。 */

const runtimeDevice = document.getElementById('runtimeDevice');
const shutdownButton = document.getElementById('shutdownButton');

fetch('/api/health')
  .then((response) => response.json())
  .then((health) => {
    if (!runtimeDevice) return;
    const runtime = health.runtime || {};
    const torchInfo = runtime.torch || {};
    if (runtime.device === 'cuda:0') {
      runtimeDevice.classList.add('gpu');
      runtimeDevice.textContent = `GPU · ${torchInfo.device_name || runtime.nvidia?.name || 'CUDA'}`;
    } else if (runtime.device === 'mps') {
      runtimeDevice.classList.add('gpu');
      runtimeDevice.textContent = `GPU · ${torchInfo.device_name || 'Apple Metal (MPS)'}`;
    } else {
      runtimeDevice.textContent = 'CPU 模式';
    }
  })
  .catch(() => {
    if (runtimeDevice) runtimeDevice.textContent = '设备状态未知';
  });

if (shutdownButton) {
  shutdownButton.addEventListener('click', async () => {
    if (!window.confirm('确定关闭蜂群视频分析程序吗？正在分析的任务也会停止。')) return;
    shutdownButton.disabled = true;
    try {
      const response = await fetch('/api/shutdown', { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || '关闭失败');
      document.body.innerHTML = `
        <main class="shutdown-screen">
          <div class="shutdown-card">
            <div class="shutdown-hex"></div>
            <h1>程序已关闭</h1>
            <p>现在可以安全关闭浏览器页面。下次使用时，请重新双击系统对应的启动文件。</p>
          </div>
        </main>`;
    } catch (error) {
      shutdownButton.disabled = false;
      window.alert(error.message || '关闭失败，请关闭命令窗口。');
    }
  });
}
