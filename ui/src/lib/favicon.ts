/**
 * Points the page's <link rel="icon"> at href, creating the link when the page has none. The
 * server changes the favicon URL's ?v= hash whenever the favicon changes, so swapping the href
 * updates the open tab's icon without a reload.
 */
export const setFavicon = (href: string, doc: Document = document) => {
  let link = doc.querySelector<HTMLLinkElement>('link[rel~="icon"]')
  if (!link) {
    link = doc.createElement('link')
    link.rel = 'icon'
    doc.head.appendChild(link)
  }
  if (link.getAttribute('href') !== href) link.setAttribute('href', href)
}

/** Favicon settings keys (see internal/instance/favicon.go). */
export const FAVICON_KEYS = {
  mode: 'instance_favicon_mode',
  color: 'instance_favicon_color',
  badge: 'instance_favicon_badge',
  upload: 'instance_favicon_upload',
  override: 'instance_favicon_override',
} as const

/** Upload limit, checked here for a quick message and again by the server. */
export const FAVICON_MAX_BYTES = 64 * 1024
export const DEFAULT_FAVICON_COLOR = '#863bff'
const UPLOAD_TYPES = ['image/png', 'image/svg+xml', 'image/x-icon', 'image/vnd.microsoft.icon']
const UPLOAD_EXT = /\.(png|svg|ico)$/i

/** Checks an upload's type and size before sending it; returns an error message or ''. */
export const checkFaviconFile = (file: File): string => {
  if (!UPLOAD_TYPES.includes(file.type) && !UPLOAD_EXT.test(file.name)) return `${file.name} is not a PNG, SVG or ICO file`
  if (file.size > FAVICON_MAX_BYTES) return `${file.name} is ${Math.ceil(file.size / 1024)} KB; the limit is 64 KB`
  return ''
}

/** Reads a file as a data URL, the form the settings API takes uploads in. */
export const readAsDataURL = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
