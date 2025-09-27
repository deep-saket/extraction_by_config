import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';
import { SafeUrlPipe } from './safe-url.pipe';

interface ResultTile { field_name: string; value: string; pages: string; }
interface FormItem {
  field_name: string; description: string; type: 'key-value' | 'bullet-points' | 'summary' | 'checkbox' | 'table';
  probable_pages: number[]; multipage_value: boolean; multiline_value: boolean; search_keys: string[];
  scope?: 'whole' | 'section' | 'pages' | 'extraction_items' | 'single_value' | 'multi_value';
  section_name?: string | null; parent?: string[]; extra?: any; table_header?: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, SafeUrlPipe],
  template: `
    <div class="container py-3">
      <div class="d-flex align-items-center justify-content-between mb-3">
        <h1 class="h4 m-0">Document Extraction</h1>
        <div class="d-flex align-items-center gap-2">
          <span class="badge bg-success" *ngIf="healthOk()">Backend: OK</span>
          <span class="badge bg-danger" *ngIf="!healthOk()">Backend: Down</span>
          <span class="badge bg-warning text-dark" *ngIf="schemaChanged()">Schema changed – reload</span>
        </div>
      </div>

      <ul class="nav nav-pills small mb-3">
        <li class="nav-item"><span class="nav-link" [class.active]="step()===1">1. Upload</span></li>
        <li class="nav-item"><span class="nav-link" [class.active]="step()===2">2. Config</span></li>
        <li class="nav-item"><span class="nav-link" [class.active]="step()===3">3. Extract</span></li>
        <li class="nav-item"><span class="nav-link" [class.active]="step()===4">4. Results</span></li>
      </ul>

      <!-- STEP 1: Upload -->
      <div *ngIf="step()===1">
        <div class="row g-3 align-items-start">
          <div class="col-12" *ngIf="!file()">
            <div class="d-flex justify-content-center py-5">
              <label class="btn btn-lg btn-primary">
                Upload PDF
                <input type="file" accept="application/pdf" hidden (change)="onFileChange($event)" />
              </label>
            </div>
          </div>
          <div class="col-12 col-lg-5" *ngIf="file()">
            <div class="mb-3">
              <label class="form-label fw-semibold">PDF File</label>
              <input class="form-control" type="file" accept="application/pdf" (change)="onFileChange($event)" />
            </div>
            <ul class="mb-3 small text-muted">
              <li>Proceed to the next step once your PDF is selected.</li>
            </ul>
            <button class="btn btn-outline-success" (click)="goToConfig()">Proceed to Config »</button>
          </div>
          <div class="col-12 col-lg-7" *ngIf="file()">
            <div class="card shadow-sm">
              <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <strong>PDF Preview</strong>
                <small class="text-muted" *ngIf="fileName()">{{ fileName() }}</small>
              </div>
              <div class="card-body p-0" style="height: 70vh;">
                <iframe [src]="pdfUrl() | safeUrl" style="width:100%;height:100%;border:0;"></iframe>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- STEP 2 and 3: PDF left, content right -->
      <div *ngIf="step()===2 || step()===3">
        <div class="row g-3">
          <div class="col-12 col-lg-7">
            <div class="card shadow-sm">
              <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <strong>PDF Preview</strong>
                <small class="text-muted" *ngIf="fileName()">{{ fileName() }}</small>
              </div>
              <div class="card-body p-0" style="height: 75vh;">
                <ng-container *ngIf="pdfUrl(); else noPdf2">
                  <iframe [src]="pdfSrc() | safeUrl" style="width:100%;height:100%;border:0;"></iframe>
                </ng-container>
                <ng-template #noPdf2>
                  <div class="d-flex align-items-center justify-content-center h-100 text-muted">No PDF selected.</div>
                </ng-template>
              </div>
            </div>
          </div>

          <div class="col-12 col-lg-5">
            <div *ngIf="step()===2" class="card shadow-sm mb-3">
              <div class="card-body">
                <div class="btn-group mb-3" role="group">
                  <button class="btn" [class.btn-primary]="configMode()==='select'" [class.btn-outline-primary]="configMode()!=='select'" (click)="configMode.set('select')">Select Config</button>
                  <button class="btn" [class.btn-primary]="configMode()==='create'" [class.btn-outline-primary]="configMode()!=='create'" (click)="configMode.set('create')">Create New</button>
                </div>

                <div *ngIf="configMode()==='select'">
                  <label class="form-label">Available Configs</label>
                  <div class="input-group mb-3">
                    <select class="form-select" [ngModel]="selectedConfig()" (ngModelChange)="onSelectConfig($event)">
                      <option *ngFor="let c of configs()" [value]="c">{{ c }}</option>
                    </select>
                    <button class="btn btn-outline-secondary" type="button" (click)="loadSelectedConfig()" [disabled]="!selectedConfig()">Load</button>
                  </div>
                </div>

                <div *ngIf="configEditorText || formMode()" class="d-flex justify-content-between align-items-center mb-2">
                  <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="formModeSwitch" [checked]="formMode()" (change)="onFormModeChange($event)">
                    <label class="form-check-label" for="formModeSwitch">Form view</label>
                  </div>
                  <small class="text-muted">Switch between Form and JSON editors</small>
                </div>

                <!-- Form Editor -->
                <div *ngIf="formMode()">
                  <div class="d-flex justify-content-between align-items-center mb-2">
                    <div class="fw-semibold">Fields ({{ formItems().length }})</div>
                    <div class="d-flex gap-2">
                      <button class="btn btn-sm btn-outline-secondary" (click)="expandAll.set(true)">Expand all</button>
                      <button class="btn btn-sm btn-outline-secondary" (click)="expandAll.set(false)">Collapse all</button>
                      <button class="btn btn-sm btn-outline-primary" (click)="addField()">Add Field</button>
                    </div>
                  </div>
                  <div class="mb-2 small text-muted text-truncate">
                    {{ fieldNamesSummary() }}
                  </div>

                  <div class="accordion" id="fieldsAcc">
                    <div class="accordion-item" *ngFor="let it of formItems(); let i = index">
                      <h2 class="accordion-header" id="h{{i}}">
                        <button class="accordion-button" [class.collapsed]="!expandAll()" type="button" data-bs-toggle="collapse" [attr.data-bs-target]="'#c'+i">
                          {{ it.field_name || 'Untitled Field' }}
                        </button>
                      </h2>
                      <div [id]="'c'+i" class="accordion-collapse collapse" [class.show]="expandAll()" [attr.aria-labelledby]="'h'+i">
                        <div class="accordion-body">
                          <div class="row g-2">
                            <div class="col-12 col-md-6">
                              <label class="form-label">Field Name</label>
                              <input class="form-control" [(ngModel)]="formItems()[i].field_name" (ngModelChange)="onFormChange()" />
                            </div>
                            <div class="col-12 col-md-6">
                              <label class="form-label">Type</label>
                              <select class="form-select" [ngModel]="formItems()[i].type" (ngModelChange)="onTypeChange(i, $event)">
                                <option value="key-value">key-value</option>
                                <option value="bullet-points">bullet-points</option>
                                <option value="summary">summary</option>
                                <option value="checkbox">checkbox</option>
                                <option value="table">table</option>
                              </select>
                            </div>
                            <div class="col-12">
                              <label class="form-label">Description</label>
                              <textarea class="form-control" rows="2" [(ngModel)]="formItems()[i].description" (ngModelChange)="onFormChange()"></textarea>
                            </div>
                            <div class="col-12 col-md-6">
                              <label class="form-label">Probable Pages (comma-separated)</label>
                              <input class="form-control" [ngModel]="formItems()[i].probable_pages?.join(', ')" (ngModelChange)="onProbablePagesChange(i, $event)" />
                            </div>
                            <div class="col-6 col-md-3 form-check mt-4">
                              <input class="form-check-input" type="checkbox" [ngModel]="formItems()[i].multipage_value" (ngModelChange)="onMultipageChange(i, $event)" id="mp{{i}}" />
                              <label class="form-check-label" [attr.for]="'mp'+i">Multipage</label>
                            </div>
                            <div class="col-6 col-md-3 form-check mt-4">
                              <input class="form-check-input" type="checkbox" [disabled]="formItems()[i].type==='table'" [ngModel]="formItems()[i].multiline_value" (ngModelChange)="onMultilineChange(i, $event)" id="ml{{i}}" />
                              <label class="form-check-label" [attr.for]="'ml'+i">Multiline</label>
                            </div>
                            <div class="col-12">
                              <label class="form-label">Search Keys (one per line)</label>
                              <textarea class="form-control" rows="2" [ngModel]="(formItems()[i].search_keys||[]).join('\\n')" (ngModelChange)="onSearchKeysChange(i, $event)"></textarea>
                            </div>

                            <div class="col-12 col-md-6" *ngIf="formItems()[i].type==='summary'">
                              <label class="form-label">Scope</label>
                              <select class="form-select" [(ngModel)]="formItems()[i].scope" (ngModelChange)="onFormChange()">
                                <option value="whole">whole</option>
                                <option value="section">section</option>
                                <option value="pages">pages</option>
                                <option value="extraction_items">extraction_items</option>
                              </select>
                            </div>
                            <div class="col-12 col-md-6" *ngIf="formItems()[i].type==='summary' && formItems()[i].scope==='section'">
                              <label class="form-label">Section Name</label>
                              <input class="form-control" [(ngModel)]="formItems()[i].section_name" (ngModelChange)="onFormChange()" />
                            </div>
                            <div class="col-12" *ngIf="formItems()[i].type==='checkbox'">
                              <label class="form-label d-block">Scope</label>
                              <div class="form-check form-check-inline">
                                <input class="form-check-input" type="radio" [name]="'chkScope'+i" [value]="'single_value'" [ngModel]="formItems()[i].scope" (ngModelChange)="onCheckboxScopeChange(i, $event)" id="chkSingle{{i}}" />
                                <label class="form-check-label" [attr.for]="'chkSingle'+i">Single value</label>
                              </div>
                              <div class="form-check form-check-inline">
                                <input class="form-check-input" type="radio" [name]="'chkScope'+i" [value]="'multi_value'" [ngModel]="formItems()[i].scope" (ngModelChange)="onCheckboxScopeChange(i, $event)" id="chkMulti{{i}}" />
                                <label class="form-check-label" [attr.for]="'chkMulti'+i">Multi value</label>
                              </div>
                            </div>

                            <div class="col-12">
                              <label class="form-label">Parent (one per line)</label>
                              <textarea class="form-control" rows="2" [ngModel]="(formItems()[i].parent||[]).join('\\n')" (ngModelChange)="onParentChange(i, $event)"></textarea>
                            </div>

                            <div class="col-12" *ngIf="formItems()[i].type==='table'">
                              <label class="form-label">Table Header (one per line; optional)</label>
                              <textarea class="form-control" rows="2" [ngModel]="(formItems()[i].table_header||[]).join('\\n')" (ngModelChange)="onTableHeaderChange(i, $event)"></textarea>
                            </div>

                            <div class="col-12">
                              <label class="form-label">Extra (JSON)</label>
                              <textarea class="form-control font-monospace" rows="3" [ngModel]="(formItems()[i].extra? stringify(formItems()[i].extra):'{}')" (ngModelChange)="onExtraChange(i, $event)"></textarea>
                            </div>

                            <div class="col-12 text-end">
                              <button class="btn btn-sm btn-outline-danger" (click)="removeField(i)">Remove</button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div class="d-flex gap-2 align-items-center mt-3">
                      <button class="btn btn-outline-primary" (click)="validateConfig()">Validate</button>
                      <button class="btn btn-outline-secondary" (click)="saveConfigAs()" [disabled]="!isConfigValid()">Save As…</button>
                      <span *ngIf="validationMsg()" class="ms-auto" [class.text-success]="isConfigValid()" [class.text-danger]="!isConfigValid()">{{ validationMsg() }}</span>
                    </div>

                    <div *ngIf="configErrors().length" class="alert alert-danger p-2 mt-2">
                      <div class="fw-semibold">Validation errors:</div>
                      <ul class="mb-0 small">
                        <li *ngFor="let e of configErrors()">{{ e.loc?.join('.') || '' }}: {{ e.msg }}</li>
                      </ul>
                    </div>

                    <div class="text-end mt-2">
                      <button class="btn btn-success" (click)="proceedToExtract()" [disabled]="!isConfigValid()">Proceed to Extraction »</button>
                    </div>
                  </div>

                  <div *ngIf="!formMode() && configEditorText">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                      <div class="fw-semibold">Config Editor</div>
                      <div class="small text-muted">Edit fields below; validate before proceeding</div>
                    </div>
                    <textarea class="form-control font-monospace mb-2" rows="10" [(ngModel)]="configEditorText"></textarea>

                    <div class="d-flex gap-2 align-items-center mb-2">
                      <button class="btn btn-outline-primary" (click)="validateConfig()">Validate</button>
                      <button class="btn btn-outline-secondary" (click)="saveConfigAs()" [disabled]="!isConfigValid()">Save As…</button>
                      <span *ngIf="validationMsg()" class="ms-auto" [class.text-success]="isConfigValid()" [class.text-danger]="!isConfigValid()">{{ validationMsg() }}</span>
                    </div>

                    <div *ngIf="configErrors().length" class="alert alert-danger p-2">
                      <div class="fw-semibold">Validation errors:</div>
                      <ul class="mb-0 small">
                        <li *ngFor="let e of configErrors()">{{ e.loc?.join('.') || '' }}: {{ e.msg }}</li>
                      </ul>
                    </div>

                    <div class="text-end">
                      <button class="btn btn-success" (click)="proceedToExtract()" [disabled]="!isConfigValid()">Proceed to Extraction »</button>
                    </div>
                  </div>

                  <div *ngIf="configMode()==='create' && !configEditorText && !formMode()">
                    <button class="btn btn-outline-primary" (click)="startNewConfig()">Start with 1 blank field</button>
                  </div>
                </div>
              </div>
            </div>

            <div *ngIf="step()===3" class="card shadow-sm">
              <div class="card-header py-2"><strong>Extracting…</strong></div>
              <div class="card-body">
                <div class="progress mb-3">
                  <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 100%"></div>
                </div>
                <div class="text-muted small">This may take a while depending on the PDF and config.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- STEP 4: Results left, PDF right -->
      <div *ngIf="step()===4">
        <div class="row g-3">
          <div class="col-12 col-lg-7">
            <div class="card shadow-sm">
              <div class="card-header py-2"><strong>Results</strong></div>
              <div class="card-body">
                <ng-container *ngIf="selectedIndex()===null; else resultDetail">
                  <div class="row g-3">
                    <div class="col-12 col-md-6" *ngFor="let t of tiles(); let i = index">
                      <button class="border rounded p-3 h-100 w-100 text-start bg-white" (click)="onTileClick(i)">
                        <div class="small text-muted">Field Name</div>
                        <div class="fw-semibold mb-1">{{ t.field_name }}</div>
                        <div class="small text-muted">Value</div>
                        <div class="mb-1" style="white-space: pre-wrap;">{{ t.value }}</div>
                        <div class="small text-muted">Pages</div>
                        <div>{{ t.pages || '-' }}</div>
                      </button>
                    </div>
                  </div>
                </ng-container>
                <ng-template #resultDetail>
                  <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="fw-semibold">Details</div>
                    <button class="btn btn-sm btn-outline-secondary" (click)="clearSelection()">Back</button>
                  </div>
                  <div class="mb-2"><span class="small text-muted">Field</span><div class="fw-semibold">{{ currentItem()?.field_name || 'Unknown' }}</div></div>
                  <div class="mb-2"><span class="small text-muted">Type</span><div>{{ currentType() }}</div></div>
                  <ng-container [ngSwitch]="currentType()">
                    <div *ngSwitchCase="'key-value'">
                      <div class="mb-2"><span class="small text-muted">Value</span><div style="white-space: pre-wrap;">{{ currentItem()?.value }}</div></div>
                      <div class="mb-2"><span class="small text-muted">Page</span><div>{{ currentItem()?.page_number }}</div></div>
                    </div>
                    <div *ngSwitchCase="'summary'">
                      <div class="mb-2"><span class="small text-muted">Summary</span><div style="white-space: pre-wrap;">{{ currentItem()?.value }}</div></div>
                      <div class="mb-2"><span class="small text-muted">Pages</span><div>{{ currentItem()?.page_range?.[0] }} - {{ currentItem()?.page_range?.[1] }}</div></div>
                    </div>
                    <div *ngSwitchCase="'bullet-points'">
                      <div class="mb-2"><span class="small text-muted">Points</span></div>
                      <ul class="mb-2">
                        <li *ngFor="let p of (currentItem()?.value || [])">{{ p.value }} <span class="text-muted small">(p{{ p.page_number }})</span></li>
                      </ul>
                    </div>
                    <div *ngSwitchCase="'checkbox'">
                      <div class="mb-2"><span class="small text-muted">Selections</span></div>
                      <ul class="mb-2">
                        <li *ngFor="let p of (currentItem()?.value || [])">{{ p.value }} <span class="text-muted small">(p{{ p.page_number }})</span></li>
                      </ul>
                    </div>
                    <div *ngSwitchCase="'table'">
                      <div class="mb-2"><span class="small text-muted">Rows</span></div>
                      <div class="table-responsive">
                        <table class="table table-sm table-bordered mb-0">
                          <tbody>
                            <tr *ngFor="let row of (currentItem()?.value || []).slice(0, 10)">
                              <td *ngFor="let cell of row">{{ cell }}</td>
                            </tr>
                          </tbody>
                        </table>
                        <div class="small text-muted mt-1">Showing up to 10 rows</div>
                      </div>
                    </div>
                    <div *ngSwitchDefault>
                      <pre class="mb-0">{{ currentItem() | json }}</pre>
                    </div>
                  </ng-container>
                </ng-template>
              </div>
            </div>
            <div class="text-end mt-3">
              <button class="btn btn-outline-secondary" (click)="resetAll()">Start Over</button>
            </div>
          </div>

          <div class="col-12 col-lg-5">
            <div class="card shadow-sm">
              <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <strong>PDF Preview</strong>
                <small class="text-muted" *ngIf="fileName()">{{ fileName() }}</small>
              </div>
              <div class="card-body p-0" style="height: 75vh;">
                <ng-container *ngIf="pdfUrl(); else noPdf3">
                  <iframe [src]="pdfSrc() | safeUrl" style="width:100%;height:100%;border:0;"></iframe>
                </ng-container>
                <ng-template #noPdf3>
                  <div class="d-flex align-items-center justify-content-center h-100 text-muted">No PDF selected.</div>
                </ng-template>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  `
})
export class AppComponent {
  step = signal<number>(1);
  configMode = signal<'select' | 'create'>('select');

  healthOk = signal(false);
  schemaHash = signal<string>('');
  lastKnownSchemaHash = signal<string>(localStorage.getItem('schema_hash') || '');

  configs = signal<string[]>([]);
  selectedConfig = signal<string>('');

  file = signal<File | null>(null);
  fileName = signal<string>('');
  pdfUrl = signal<string>('');

  result = signal<any>(null);
  tiles = signal<ResultTile[]>([]);

  configEditorText = '';
  configErrors = signal<any[]>([]);
  isConfigValid = signal<boolean>(false);
  validationMsg = signal<string>('');
  formMode = signal<boolean>(false);
  formItems = signal<FormItem[]>([]);

  pdfPage = signal<number | null>(null);
  selectedIndex = signal<number | null>(null);
  expandAll = signal<boolean>(true);

  constructor(private api: ApiService) { this.init(); }

  private init() {
    this.api.health().subscribe({ next: () => this.healthOk.set(true), error: () => this.healthOk.set(false) });
    this.api.listConfigs().subscribe({ next: r => { this.configs.set(r.configs); if (!this.selectedConfig() && r.configs.length) this.selectedConfig.set(r.configs[0]); } });
    this.api.schemaVersion().subscribe({ next: r => { this.schemaHash.set(r.schema_hash); const prev = this.lastKnownSchemaHash(); if (!prev) { localStorage.setItem('schema_hash', r.schema_hash); this.lastKnownSchemaHash.set(r.schema_hash); } } });
  }

  schemaChanged(): boolean { return !!this.lastKnownSchemaHash() && !!this.schemaHash() && this.schemaHash() !== this.lastKnownSchemaHash(); }

  onFileChange(ev: Event) { const input = ev.target as HTMLInputElement; if (input.files && input.files.length > 0) { const f = input.files[0]; this.file.set(f); this.fileName.set(f.name); this.pdfUrl.set(URL.createObjectURL(f)); } }
  goToConfig() { if (this.file()) this.step.set(2); }
  onSelectConfig(name: string) { this.selectedConfig.set(name); }

  loadSelectedConfig() {
    const cfg = this.selectedConfig(); if (!cfg) return;
    this.api.getConfig(cfg).subscribe({ next: data => { this.configEditorText = JSON.stringify(data, null, 2); this.formItems.set(this.parseToForm(data)); this.isConfigValid.set(false); this.configErrors.set([]); this.validationMsg.set('Loaded. Validate to check.'); this.formMode.set(false); }, error: () => this.validationMsg.set('Failed to load config') });
  }

  startNewConfig() {
    const blank: FormItem[] = [{ field_name: 'Field1', description: 'Describe this field', type: 'key-value', probable_pages: [], multipage_value: false, multiline_value: false, search_keys: [], scope: undefined, section_name: undefined, parent: [], extra: {}, table_header: [] }];
    this.formItems.set(blank);
    this.configEditorText = JSON.stringify(this.formItems(), null, 2);
    this.isConfigValid.set(false);
    this.configErrors.set([]);
    this.validationMsg.set('Draft created. Validate to check.');
    this.formMode.set(true);
  }

  validateConfig() {
    if (this.formMode()) this.configEditorText = JSON.stringify(this.formItems(), null, 2);
    let parsed: any; try { parsed = JSON.parse(this.configEditorText || ''); } catch { this.isConfigValid.set(false); this.configErrors.set([{ msg: 'Invalid JSON', loc: ['json'] }]); this.validationMsg.set('Invalid JSON'); return; }
    this.api.validateConfig(parsed).subscribe({ next: r => { if (r.ok) { this.isConfigValid.set(true); this.configErrors.set([]); this.validationMsg.set('Config valid'); } else { this.isConfigValid.set(false); this.configErrors.set(r.errors || []); this.validationMsg.set('Config invalid'); } }, error: e => { this.isConfigValid.set(false); const errs = e?.error?.errors || [{ msg: 'Validation failed', loc: [] }]; this.configErrors.set(errs); this.validationMsg.set('Config invalid'); } });
  }

  saveConfigAs() {
    if (!this.isConfigValid()) return;
    const name = window.prompt('Save config as (filename.json):', this.selectedConfig() || 'custom.json'); if (!name) return;
    let parsed: any; try { parsed = JSON.parse(this.configEditorText || ''); } catch { return; }
    this.api.saveConfig(name, parsed).subscribe({ next: r => { if (r.ok) { this.validationMsg.set(`Saved as ${name}`); this.api.listConfigs().subscribe({ next: lst => this.configs.set(lst.configs) }); } else { this.validationMsg.set('Save failed'); } }, error: () => this.validationMsg.set('Save failed') });
  }

  proceedToExtract() {
    this.step.set(3);
    const f = this.file(); if (!f) return;
    let useInline = this.isConfigValid();
    if (useInline) {
      if (this.formMode()) this.configEditorText = JSON.stringify(this.formItems(), null, 2);
      let parsed: any; try { parsed = JSON.parse(this.configEditorText || ''); } catch { useInline = false; }
      if (useInline) { this.api.performDEWithInlineConfig(f, parsed).subscribe({ next: r => this.onExtractionDone(r), error: e => this.onExtractionError(e) }); return; }
    }
    const cfg = this.selectedConfig(); if (!cfg) { this.onExtractionError({ message: 'No config selected' }); return; }
    this.api.performDE(f, cfg).subscribe({ next: r => this.onExtractionDone(r), error: e => this.onExtractionError(e) });
  }

  private onExtractionDone(r: any) { this.result.set(r); this.tiles.set(this.normalizeResults(r)); this.step.set(4); }
  private onExtractionError(e: any) { this.result.set({ error: e?.message || 'Extraction failed' }); this.tiles.set([]); this.step.set(4); }

  private normalizeResults(r: any): ResultTile[] {
    if (!Array.isArray(r)) return [];
    const tiles: ResultTile[] = [];
    for (const item of r) {
      const typeGuess = this.guessType(item);
      const field = item.field_name || item?.root?.field_name || 'Unknown';
      let value = ''; let pages = '';
      switch (typeGuess) {
        case 'key-value': value = String(item.value ?? ''); pages = String(item.page_number ?? ''); break;
        case 'summary': value = String(item.value ?? ''); if (Array.isArray(item.page_range) && item.page_range.length === 2) pages = `${item.page_range[0]}-${item.page_range[1]}`; break;
        case 'bullet-points':
        case 'checkbox':
          if (Array.isArray(item.value)) {
            value = item.value.map((p: any) => p?.value).filter(Boolean).join('\n• ');
            const pgSet = new Set<number>(); for (const p of item.value) if (p?.page_number) pgSet.add(p.page_number);
            pages = Array.from(pgSet).sort((a,b)=>a-b).join(', ');
            if (value) value = '• ' + value;
          }
          break;
        case 'table':
          const rows = Array.isArray(item.value) ? item.value : [];
          value = `${rows.length} rows`;
          if (Array.isArray(item.page_numbers)) pages = item.page_numbers.join(', ');
          break;
        default: value = typeof item.value === 'object' ? JSON.stringify(item.value) : String(item.value ?? '');
      }
      tiles.push({ field_name: field, value, pages });
    }
    return tiles;
  }

  private guessType(item: any): string {
    if (typeof item?.page_number === 'number' && typeof item?.value === 'string') return 'key-value';
    if (Array.isArray(item?.value) && item?.value.every((v: any) => v && typeof v === 'object' && 'index' in v && 'page_number' in v)) return 'bullet-points';
    if (Array.isArray(item?.page_numbers) && Array.isArray(item?.value)) return 'table';
    if (typeof item?.value === 'string' && ('page_range' in item || 'related_fields' in item)) return 'summary';
    return 'unknown';
  }

  resetAll() {
    const url = this.pdfUrl(); if (url) URL.revokeObjectURL(url);
    this.step.set(1); this.configMode.set('select');
    this.file.set(null); this.fileName.set(''); this.pdfUrl.set('');
    this.configEditorText = ''; this.isConfigValid.set(false); this.configErrors.set([]); this.validationMsg.set('');
    this.result.set(null); this.tiles.set([]);
  }

  parseToForm(data: any): FormItem[] {
    try {
      const arr: any[] = Array.isArray(data) ? data : JSON.parse(this.configEditorText || '[]');
      return arr.map((it: any) => ({
        field_name: String(it.field_name || ''), description: String(it.description || ''), type: (it.type || 'key-value'),
        probable_pages: Array.isArray(it.probable_pages) ? it.probable_pages.map((x: any) => Number(x)).filter((n: any) => !isNaN(n)) : [],
        multipage_value: !!it.multipage_value, multiline_value: !!it.multiline_value,
        search_keys: Array.isArray(it.search_keys) ? it.search_keys.map((s: any) => String(s)) : [],
        scope: it.scope, section_name: it.section_name,
        parent: Array.isArray(it.parent) ? it.parent.map((s: any) => String(s)) : [],
        extra: it.extra ?? {}, table_header: Array.isArray(it.table_header) ? it.table_header.map((s: any) => String(s)) : []
      } as FormItem));
    } catch { return []; }
  }

  addField() {
    const arr = [...this.formItems()];
    arr.push({ field_name: 'NewField', description: '', type: 'key-value', probable_pages: [], multipage_value: false, multiline_value: false, search_keys: [], scope: undefined, section_name: undefined, parent: [], extra: {}, table_header: [] });
    this.formItems.set(arr); this.configEditorText = JSON.stringify(arr, null, 2);
  }

  removeField(idx: number) { const arr = [...this.formItems()]; arr.splice(idx, 1); this.formItems.set(arr); this.configEditorText = JSON.stringify(arr, null, 2); }
  onFormChange() { this.configEditorText = JSON.stringify(this.formItems(), null, 2); this.isConfigValid.set(false); this.validationMsg.set('Edited. Validate to check.'); }
  onTypeChange(i: number, newType: FormItem['type']) { this.formItems()[i].type = newType; if (newType === 'table') this.formItems()[i].multiline_value = true; this.onFormChange(); }
  onFormModeChange(evt: Event) {
    const checked = (evt.target as HTMLInputElement).checked;
    this.formMode.set(checked);
    if (checked) {
      this.formItems.set(this.parseToForm(null));
    } else {
      this.configEditorText = JSON.stringify(this.formItems(), null, 2);
    }
  }
  onExtraChange(i: number, text: string) {
    try { this.formItems()[i].extra = JSON.parse(text || '{}'); }
    catch { /* keep previous extra */ }
    this.onFormChange();
  }
  onProbablePagesChange(i: number, text: string) {
    const arr = (text || '').split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
    this.formItems()[i].probable_pages = arr;
    this.onFormChange();
  }
  onMultipageChange(i: number, val: any) {
    this.formItems()[i].multipage_value = !!val;
    this.onFormChange();
  }
  onMultilineChange(i: number, val: any) {
    if (this.formItems()[i].type === 'table') return; // enforced true elsewhere
    this.formItems()[i].multiline_value = !!val;
    this.onFormChange();
  }
  onSearchKeysChange(i: number, text: string) {
    this.formItems()[i].search_keys = (text || '').split('\n').map(s => s.trim()).filter(Boolean);
    this.onFormChange();
  }
  onParentChange(i: number, text: string) {
    this.formItems()[i].parent = (text || '').split('\n').map(s => s.trim()).filter(Boolean);
    this.onFormChange();
  }
  onTableHeaderChange(i: number, text: string) {
    this.formItems()[i].table_header = (text || '').split('\n').map(s => s.trim()).filter(Boolean);
    this.onFormChange();
  }
  onCheckboxScopeChange(i: number, val: 'single_value' | 'multi_value') {
    this.formItems()[i].scope = val;
    this.onFormChange();
  }

  pdfSrc(): string {
    const base = this.pdfUrl();
    if (!base) return '';
    const p = this.pdfPage();
    return p ? `${base}#page=${p}` : base;
  }
  onTileClick(i: number) {
    const items = Array.isArray(this.result()) ? this.result() : [];
    const item = items[i];
    this.selectedIndex.set(i);
    const page = this.firstPageOf(item);
    if (page) this.pdfPage.set(page);
  }
  clearSelection() { this.selectedIndex.set(null); }
  currentItem(): any { const idx = this.selectedIndex(); return idx===null ? null : (this.result() || [])[idx]; }
  currentType(): string { return this.guessType(this.currentItem()); }
  fieldNamesSummary(): string {
    try {
      return (this.formItems() || []).map((it: any) => it?.field_name || 'Untitled').join(', ');
    } catch { return ''; }
  }

  private firstPageOf(item: any): number | null {
    if (!item) return null;
    const t = this.guessType(item);
    if (t === 'key-value') {
      return typeof item.page_number === 'number' ? item.page_number : null;
    }
    if (t === 'summary') {
      return Array.isArray(item.page_range) && item.page_range.length ? Number(item.page_range[0]) : null;
    }
    if (t === 'bullet-points' || t === 'checkbox') {
      if (Array.isArray(item.value)) {
        const nums = item.value
          .map((v: any) => v?.page_number)
          .filter((n: any) => typeof n === 'number');
        return nums.length ? Math.min(...nums) : null;
      }
      return null;
    }
    if (t === 'table') {
      return Array.isArray(item.page_numbers) && item.page_numbers.length ? Number(item.page_numbers[0]) : null;
    }
    return null;
  }

  stringify(val: any): string {
    try { return JSON.stringify(val, null, 2); } catch { return '{}'; }
  }
}
