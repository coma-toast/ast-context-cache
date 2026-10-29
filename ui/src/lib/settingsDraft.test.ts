import { describe, expect, it } from 'vitest'

import { changedKeys, draftChanges, draftValue, editsAfterSave, SECRET_CLEARED, SECRET_SET } from './settingsDraft'

const secrets = new Set(['TOKEN'])

describe('settingsDraft', () => {
  it.each([
    ['unchanged edit', { a: '1' }, { a: '1' }, {}, []],
    ['changed edit', { a: '1' }, { a: '2' }, {}, ['a']],
    ['edit matching a committed save', { a: '1' }, { a: '2' }, { a: '2' }, []],
    ['blank secret is unchanged', { TOKEN: SECRET_SET }, { TOKEN: '' }, {}, []],
    ['typed secret', { TOKEN: SECRET_SET }, { TOKEN: 'new' }, {}, ['TOKEN']],
    ['cleared set secret', { TOKEN: SECRET_SET }, { TOKEN: SECRET_CLEARED }, {}, ['TOKEN']],
    ['cleared unset secret', { TOKEN: '' }, { TOKEN: SECRET_CLEARED }, {}, []],
  ])('changedKeys: %s', (_name, saved, edits, committed, want) => {
    expect(changedKeys(saved, edits, committed, secrets)).toEqual(want)
  })

  it('shows edits over commits over saved values, and never a stored secret', () => {
    expect(draftValue('a', { a: '1' }, {}, {}, secrets)).toBe('1')
    expect(draftValue('a', { a: '1' }, {}, { a: '2' }, secrets)).toBe('2')
    expect(draftValue('a', { a: '1' }, { a: '3' }, { a: '2' }, secrets)).toBe('3')
    expect(draftValue('TOKEN', { TOKEN: SECRET_SET }, {}, {}, secrets)).toBe('')
    expect(draftValue('TOKEN', { TOKEN: SECRET_SET }, { TOKEN: SECRET_CLEARED }, {}, secrets)).toBe('')
  })

  it('sends a cleared secret as empty', () => {
    expect(draftChanges(['TOKEN', 'a'], { TOKEN: SECRET_CLEARED, a: '2' })).toEqual({ TOKEN: '', a: '2' })
  })

  it('keeps only edits made while the save was in flight', () => {
    expect(editsAfterSave({ a: '2', b: '5', TOKEN: SECRET_CLEARED }, { a: '2', b: '4', TOKEN: '' })).toEqual({ b: '5' })
  })
})
