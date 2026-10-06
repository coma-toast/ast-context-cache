import { api } from '../api/client'

type ShowToast = (message: string, severity?: 'success' | 'error' | 'info') => void

// restartNow asks ast-mcp to restart itself; shared with the Storage card. The
// restart (drain + exec) is a separate, explicit action rather than automatic; see
// the backend's db.RestartProcess doc comment for why.
export async function restartNow(showToast: ShowToast, setRestarting: (v: boolean) => void) {
  setRestarting(true)
  try {
    await api.restartNow()
    showToast('Restarting ast-mcp — this page will reconnect once it\'s back', 'success')
  } catch (e) {
    showToast(String(e), 'error')
    setRestarting(false)
  }
}
