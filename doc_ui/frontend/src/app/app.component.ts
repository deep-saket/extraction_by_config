import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';
import { SafeUrlPipe } from './safe-url.pipe';
import { HttpClientModule } from '@angular/common/http';

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
  imports: [CommonModule, FormsModule, SafeUrlPipe, HttpClientModule],
  template: `
    <div class="container py-3">
      <div class="d-flex align-items-center justify-content-between mb-3">
        <h1 class="h4 m-0">Document Extraction</h1>
        <div class="d-flex align-items-center gap-2">
          <span class="badge bg-success" *ngIf="healthOk()">Backend: OK</span>
          <span class="badge bg-danger" *ngIf="!healthOk()">Backend: Down</span>
          <span class="badge bg-warning text-dark" *ngIf="schemaChanged()">Schema changed – reload</span>
          <button class="btn btn-sm btn-outline-secondary" (click)="loadLocalOutput()">Load demo output</button>
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
            <div class="card shadow-sm sticky-top" style="top: 12px;">
              <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <strong>PDF Preview</strong>
                <div class="d-flex align-items-center gap-3">
                  <div class="form-check form-switch m-0 p-0 d-flex align-items-center gap-1">
                    <input class="form-check-input" type="checkbox" id="lockSwitchA" [checked]="pdfLocked()" (change)="onPdfLockToggle($event)">
                    <label class="form-check-label small" for="lockSwitchA">Lock scroll</label>
                  </div>
                  <small class="text-muted" *ngIf="fileName()">{{ fileName() }}</small>
                </div>
              </div>
              <div class="card-body p-0" style="height: 75vh;" [style.pointer-events]="pdfLocked() ? 'none' : 'auto'">
                <ng-container *ngIf="pdfUrl() && pdfIframeVisible(); else noPdf2">
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

                <div *ngIf="configMode()==='create'" class="border rounded p-3 mb-3 bg-light-subtle">
                  <div class="d-flex flex-column flex-lg-row align-items-lg-end justify-content-between gap-3">
                    <div>
                      <div class="fw-semibold">Auto-generate config</div>
                      <div class="small text-muted">Let the VLM inspect the PDF and propose ExtractionItems (optional).</div>
                    </div>
                    <div class="d-flex flex-wrap gap-2 align-items-end">
                      <div class="small">
                        <label class="form-label small mb-1">Max pages (optional)</label>
                        <input type="number" min="1" class="form-control form-control-sm" [ngModel]="autoConfigMaxPages()" (ngModelChange)="onAutoConfigMaxPagesChange($event)" placeholder="All" />
                      </div>
                      <button class="btn btn-sm btn-primary" type="button" (click)="generateAutoConfig()" [disabled]="autoConfigBusy() || !file()">Generate</button>
                    </div>
                  </div>
                  <div class="mt-2 small">
                    <span *ngIf="autoConfigBusy()" class="text-muted">Generating config…</span>
                    <span *ngIf="!autoConfigBusy() && autoConfigMsg()" class="text-success">{{ autoConfigMsg() }}</span>
                    <span *ngIf="!autoConfigBusy() && autoConfigError()" class="text-danger">{{ autoConfigError() }}</span>
                    <span *ngIf="!file()" class="text-muted d-block">Upload a PDF to enable auto-generation.</span>
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
                      <button class="btn btn-outline-success" type="button" (click)="exportConfig()" [disabled]="!isConfigValid()">Export & Use</button>
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
                      <button class="btn btn-outline-success" type="button" (click)="exportConfig()" [disabled]="!isConfigValid()">Export & Use</button>
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
                  <!-- Drill-down view -->
                  <ng-container *ngIf="drilledParentIndex()!==null; else mainGrid">
                    <div class="d-flex align-items-center justify-content-between mb-2">
                      <div class="fw-semibold">Entries for {{ getItemAt(drilledParentIndex()!)?.field_name || 'Unknown' }}</div>
                      <button class="btn btn-sm btn-outline-secondary" (click)="exitDrillDown()">Back</button>
                    </div>
                    <div class="row g-3">
                      <div class="col-12 col-md-6" *ngFor="let e of drilledEntries(); let j = index">
                        <div class="border rounded p-3 h-100 w-100 bg-white">
                          <div class="small text-muted">Entry {{ j+1 }}</div>
                          <div style="white-space: pre-wrap;">{{ e.label }}</div>
                          <div class="mt-2" *ngIf="e.page">
                            <button type="button" class="btn btn-sm btn-light border" (click)="gotoPage(e.page)">Go to page {{ e.page }}</button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </ng-container>
                  <!-- Main tiles grid -->
                  <ng-template #mainGrid>
                    <div class="row g-3">
                      <div class="col-12 col-md-6" *ngFor="let t of tiles(); let i = index">
                        <div role="button" class="border rounded p-3 h-100 w-100 text-start bg-white" style="cursor: pointer;" (click)="onTileAreaClick(i)">
                          <div class="small text-muted">Field Name</div>
                          <div class="fw-semibold mb-1">{{ t.field_name }}</div>
                          <div class="small text-muted">Value</div>
                          <div class="mb-1" style="white-space: pre-wrap;">
                            {{ expandedTileIndex()===i ? tileFullText(getItemAt(i)) : tilePreviewText(getItemAt(i)) }}
                          </div>
                          <div class="d-flex gap-2 mb-2">
                            <a href="#" (click)="$event.preventDefault(); onTileClick(i)">View details</a>
                            <a href="#" *ngIf="isMultiEntry(getItemAt(i)) && expandedTileIndex()===i" (click)="$event.preventDefault(); drilledParentIndex.set(i)">See entries</a>
                          </div>
                          <div class="small text-muted">Pages</div>
                          <div *ngIf="pagesForIndex(i).length" class="d-flex flex-wrap gap-1">
                            <button type="button" class="btn btn-sm btn-light border" *ngFor="let p of pagesForIndex(i)" (click)="gotoPage(p, $event)">{{ p }}</button>
                          </div>
                          <div *ngIf="!pagesForIndex(i).length">-</div>
                        </div>
                      </div>
                    </div>
                  </ng-template>
                 </ng-container>
                <ng-template #resultDetail>
                  <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="fw-semibold">Details</div>
                    <button class="btn btn-sm btn-outline-secondary" (click)="clearSelection()">Back</button>
                  </div>
                  <div class="mb-2"><span class="small text-muted">Field</span><div class="fw-semibold">{{ currentItem()?.field_name || 'Unknown' }}</div></div>
                  <div class="mb-2"><span class="small text-muted">Type</span><div>{{ currentType() }}</div></div>
                  <div class="mb-2" *ngIf="pagesForSelected().length">
                    <span class="small text-muted">Pages</span>
                    <div class="d-flex flex-wrap gap-1 mt-1">
                      <button type="button" class="btn btn-sm btn-light border" *ngFor="let p of pagesForSelected()" (click)="gotoPage(p)">{{ p }}</button>
                    </div>
                  </div>
                  <ng-container [ngSwitch]="currentType()">
                    <div *ngSwitchCase="'key-value'">
                       <div class="mb-2"><span class="small text-muted">Value</span><div style="white-space: pre-wrap;">{{ currentItem()?.value }}</div></div>
                       <div class="mb-2" *ngIf="currentItem()?.post_processing_value">
                         <span class="small text-muted">Post-processed</span>
                         <div style="white-space: pre-wrap;">{{ currentItem()?.post_processing_value }}</div>
                       </div>
                       <div class="mb-2"><span class="small text-muted">Key</span><div>{{ currentItem()?.key }}</div></div>
                       <div class="mb-2" *ngIf="(currentItem()?.multipage_detail||[]).length">
                         <div class="small text-muted">Fragments</div>
                         <ul class="mb-2">
                           <li *ngFor="let f of (currentItem()?.multipage_detail||[])">
                             <button type="button" class="btn btn-xs btn-light border py-0 px-1 me-2" (click)="gotoPage(f.page_number)">p{{ f.page_number }}</button>
                             <span>{{ f.value }}</span>
                             <span class="text-muted small" *ngIf="f.post_processing_value"> → {{ f.post_processing_value }}</span>
                           </li>
                         </ul>
                       </div>
                    </div>
                    <div *ngSwitchCase="'summary'">
                       <div class="mb-2"><span class="small text-muted">Summary</span><div style="white-space: pre-wrap;">{{ currentItem()?.value }}</div></div>
                       <div class="mb-2" *ngIf="(currentItem()?.related_fields||[]).length">
                         <span class="small text-muted">Related fields</span>
                         <div class="mt-1 d-flex flex-wrap gap-1">
                           <span class="badge bg-light text-dark border" *ngFor="let rf of currentItem()?.related_fields">{{ rf }}</span>
                         </div>
                       </div>
                       <div class="mb-2"><span class="small text-muted">Key</span><div>{{ currentItem()?.key }}</div></div>
                    </div>
                    <div *ngSwitchCase="'bullet-points'">
                      <div class="mb-2"><span class="small text-muted">Points</span></div>
                      <ul class="mb-2">
                        <li *ngFor="let p of (currentItem()?.value || [])">
                          <button type="button" class="btn btn-xs btn-light border py-0 px-1 me-2" (click)="gotoPage(p.page_number)">p{{ p.page_number }}</button>
                          <span>{{ p.value }}</span>
                          <span class="text-muted small" *ngIf="p.post_processing_value"> → {{ p.post_processing_value }}</span>
                        </li>
                      </ul>
                      <div class="mb-2"><span class="small text-muted">Key</span><div>{{ currentItem()?.key }}</div></div>
                    </div>
                    <div *ngSwitchCase="'checkbox'">
                      <div class="mb-2"><span class="small text-muted">Selections</span></div>
                      <ul class="mb-2">
                        <li *ngFor="let p of (currentItem()?.value || [])">
                          <button type="button" class="btn btn-xs btn-light border py-0 px-1 me-2" (click)="gotoPage(p.page_number)">p{{ p.page_number }}</button>
                          <span>{{ p.value }}</span>
                          <span class="text-muted small" *ngIf="p.post_processing_value"> → {{ p.post_processing_value }}</span>
                        </li>
                      </ul>
                      <div class="mb-2"><span class="small text-muted">Key</span><div>{{ currentItem()?.key }}</div></div>
                    </div>
                    <div *ngSwitchCase="'table'">
                      <div class="mb-2 d-flex justify-content-between align-items-center">
                        <div><span class="small text-muted">Table</span></div>
                        <div class="d-flex align-items-center gap-2">
                          <div class="btn-group btn-group-sm" role="group">
                            <button class="btn btn-outline-secondary" (click)="prevTablePage()" [disabled]="tablePageIndex()===0">Prev</button>
                            <button class="btn btn-outline-secondary" (click)="nextTablePage()" [disabled]="tablePageIndex() >= totalTablePages()-1">Next</button>
                          </div>
                          <div class="small text-muted">Page {{ tablePageIndex()+1 }} / {{ totalTablePages() }}</div>
                          <select class="form-select form-select-sm" style="width: auto;" [ngModel]="tablePageSize()" (ngModelChange)="onTablePageSizeChange($event)">
                            <option [value]="5">5</option>
                            <option [value]="10">10</option>
                            <option [value]="25">25</option>
                            <option [value]="50">50</option>
                          </select>
                          <button class="btn btn-sm btn-outline-primary" (click)="exportTableCSV()">Export CSV</button>
                        </div>
                      </div>
                      
                      <div class="table-responsive">
                        <table class="table table-sm table-bordered mb-0">
                          <thead>
                            <tr>
                              <th *ngFor="let h of tableHeaders">{{ h }}</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr *ngFor="let rv of paginatedTableRows()">
                              <td *ngFor="let cell of rv">{{ cell }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                      
                      <div class="mb-2" *ngIf="(currentItem()?.multipage_detail||[]).length">
                        <span class="small text-muted">Row fragments</span>
                        <ul class="mb-2">
                          <li *ngFor="let rf of (currentItem()?.multipage_detail||[])">
                            <button type="button" class="btn btn-xs btn-light border py-0 px-1 me-2" (click)="gotoPage(rf.page_number)">p{{ rf.page_number }}</button>
                            <span>#{{ rf.index }}</span>
                          </li>
                        </ul>
                      </div>
                      <div class="mb-2"><span class="small text-muted">Key</span><div>{{ currentItem()?.key }}</div></div>
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
            <div class="card shadow-sm sticky-top" style="top: 12px;">
               <div class="card-header py-2 d-flex justify-content-between align-items-center">
                 <strong>PDF Preview</strong>
                 <div class="d-flex align-items-center gap-3">
                  <div class="form-check form-switch m-0 p-0 d-flex align-items-center gap-1">
                    <input class="form-check-input" type="checkbox" id="lockSwitchB" [checked]="pdfLocked()" (change)="onPdfLockToggle($event)">
                    <label class="form-check-label small" for="lockSwitchB">Lock scroll</label>
                  </div>
                  <small class="text-muted" *ngIf="fileName()">{{ fileName() }}</small>
                </div>
               </div>
               <div class="card-body p-0" style="height: 75vh;" [style.pointer-events]="pdfLocked() ? 'none' : 'auto'">
                 <ng-container *ngIf="pdfUrl() && pdfIframeVisible(); else noPdf3">
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
  pdfLocked = signal<boolean>(true);

  result = signal<any>(null);
  tiles = signal<ResultTile[]>([]);

  configEditorText = '';
  configErrors = signal<any[]>([]);
  isConfigValid = signal<boolean>(false);
  validationMsg = signal<string>('');
  formMode = signal<boolean>(false);
  formItems = signal<FormItem[]>([]);
  autoConfigBusy = signal<boolean>(false);
  autoConfigMsg = signal<string>('');
  autoConfigError = signal<string>('');
  autoConfigMaxPages = signal<number | null>(null);

  pdfPage = signal<number | null>(null);
  selectedIndex = signal<number | null>(null);
  expandAll = signal<boolean>(true);
  pdfIframeVisible = signal<boolean>(true);
  navVersion = signal<number>(0);
  expandedTileIndex = signal<number | null>(null);
  drilledParentIndex = signal<number | null>(null);
  // Pagination state for detailed table view
  tablePageSize = signal<number>(10);
  tablePageIndex = signal<number>(0);
  inlineExtractionPreferred = signal<boolean>(true);

  constructor(private api: ApiService) { this.init(); }

  private init() {
    this.api.health().subscribe({ next: () => this.healthOk.set(true), error: () => this.healthOk.set(false) });
    this.api.listConfigs().subscribe({ next: r => { this.configs.set(r.configs); if (!this.selectedConfig() && r.configs.length) this.selectedConfig.set(r.configs[0]); } });
    this.api.schemaVersion().subscribe({ next: r => { this.schemaHash.set(r.schema_hash); const prev = this.lastKnownSchemaHash(); if (!prev) { localStorage.setItem('schema_hash', r.schema_hash); this.lastKnownSchemaHash.set(r.schema_hash); } } });
  }

  schemaChanged(): boolean { return !!this.lastKnownSchemaHash() && !!this.schemaHash() && this.schemaHash() !== this.lastKnownSchemaHash(); }

  onFileChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const f = input.files[0];
      this.file.set(f);
      this.fileName.set(f.name);
      this.pdfUrl.set(URL.createObjectURL(f));
      this.pdfPage.set(null);
      this.navVersion.set(0);
      this.pdfIframeVisible.set(true);
      this.expandedTileIndex.set(null);
      this.drilledParentIndex.set(null);
      this.autoConfigBusy.set(false);
      this.autoConfigMsg.set('');
      this.autoConfigError.set('');
      this.inlineExtractionPreferred.set(true);
    }
  }
  goToConfig() { if (this.file()) this.step.set(2); }
  onSelectConfig(name: string) { this.selectedConfig.set(name); }

  loadSelectedConfig() {
    const cfg = this.selectedConfig(); if (!cfg) return;
    this.api.getConfig(cfg).subscribe({ next: data => { this.configEditorText = JSON.stringify(data, null, 2); this.formItems.set(this.parseToForm(data)); this.isConfigValid.set(false); this.configErrors.set([]); this.validationMsg.set('Loaded. Validate to check.'); this.formMode.set(false); this.inlineExtractionPreferred.set(false); }, error: () => this.validationMsg.set('Failed to load config') });
  }

  startNewConfig() {
    const blank: FormItem[] = [{ field_name: 'Field1', description: 'Describe this field', type: 'key-value', probable_pages: [], multipage_value: false, multiline_value: false, search_keys: [], scope: undefined, section_name: undefined, parent: [], extra: {}, table_header: [] }];
    this.formItems.set(blank);
    this.configEditorText = JSON.stringify(this.formItems(), null, 2);
    this.isConfigValid.set(false);
    this.configErrors.set([]);
    this.validationMsg.set('Draft created. Validate to check.');
    this.formMode.set(true);
    this.inlineExtractionPreferred.set(true);
  }

  generateAutoConfig() {
    const pdfFile = this.file();
    if (!pdfFile) {
      this.autoConfigError.set('Upload a PDF first to enable auto-generation.');
      return;
    }
    this.autoConfigBusy.set(true);
    this.autoConfigMsg.set('');
    this.autoConfigError.set('');

    const maxPages = this.autoConfigMaxPages();
    this.api.generateAutoConfig(pdfFile, maxPages ?? undefined).subscribe({
      next: res => {
        const items = Array.isArray(res?.items) ? res.items : [];
        if (!items.length) {
          this.autoConfigError.set('No fields detected. Try increasing the page range or adjust manually.');
          this.autoConfigBusy.set(false);
          return;
        }
        this.configMode.set('create');
        this.configEditorText = JSON.stringify(items, null, 2);
        this.formItems.set(this.parseToForm(items));
        this.formMode.set(true);
        this.isConfigValid.set(true);
        this.configErrors.set([]);
        this.validationMsg.set('Auto-generated config ready. Review and adjust as needed.');
        this.autoConfigMsg.set(`Generated ${items.length} fields.`);
        this.inlineExtractionPreferred.set(true);
        this.autoConfigBusy.set(false);
      },
      error: err => {
        const message = err?.error?.error || err?.message || 'Failed to generate config.';
        this.autoConfigError.set(message);
        this.autoConfigBusy.set(false);
      }
    });
  }

  onAutoConfigMaxPagesChange(value: any) {
    if (value === null || value === undefined || value === '') {
      this.autoConfigMaxPages.set(null);
      return;
    }
    const num = Number(value);
    if (isNaN(num) || num <= 0) {
      this.autoConfigMaxPages.set(null);
      return;
    }
    this.autoConfigMaxPages.set(Math.floor(num));
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

  exportConfig() {
    if (this.formMode()) this.configEditorText = JSON.stringify(this.formItems(), null, 2);
    let parsed: any;
    try { parsed = JSON.parse(this.configEditorText || ''); }
    catch { this.isConfigValid.set(false); this.validationMsg.set('Invalid JSON'); this.inlineExtractionPreferred.set(true); return; }

    const filename = this.buildExportFileName();
    this.api.saveConfig(filename, parsed).subscribe({
      next: res => {
        if (res.ok) {
          this.validationMsg.set(`Exported as ${filename}. Using saved config for extraction.`);
          this.inlineExtractionPreferred.set(false);
          this.selectedConfig.set(filename);
          this.configMode.set('select');
          this.configErrors.set([]);
          this.api.listConfigs().subscribe({
            next: lst => {
              const configs = lst.configs.includes(filename) ? lst.configs : [...lst.configs, filename];
              const unique = Array.from(new Set(configs));
              this.configs.set(unique.sort());
            },
          });
        } else {
          this.validationMsg.set('Export failed');
        }
      },
      error: () => this.validationMsg.set('Export failed')
    });
  }

  proceedToExtract() {
    this.step.set(3);
    const f = this.file(); if (!f) return;
    let useInline = this.inlineExtractionPreferred() && this.isConfigValid();
    if (useInline) {
      if (this.formMode()) this.configEditorText = JSON.stringify(this.formItems(), null, 2);
      let parsed: any; try { parsed = JSON.parse(this.configEditorText || ''); } catch { useInline = false; }
      if (!useInline) this.inlineExtractionPreferred.set(false);
      if (useInline) { this.api.performDEWithInlineConfig(f, parsed).subscribe({ next: r => this.onExtractionDone(r), error: e => this.onExtractionError(e) }); return; }
    }
    const cfg = this.selectedConfig(); if (!cfg) { this.onExtractionError({ message: 'No config selected' }); return; }
    this.inlineExtractionPreferred.set(false);
    this.api.performDE(f, cfg).subscribe({ next: r => this.onExtractionDone(r), error: e => this.onExtractionError(e) });
  }

  private onExtractionDone(r: any) { this.result.set(r); this.tiles.set(this.normalizeResults(r)); this.step.set(4); }
  private onExtractionError(e: any) { this.result.set({ error: e?.message || 'Extraction failed' }); this.tiles.set([]); this.step.set(4); }

  private buildExportFileName(): string {
    const pdf = this.file();
    const rawBase = pdf?.name ? pdf.name.replace(/\.[^.]+$/, '') : 'auto_config';
    const slug = rawBase.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'auto_config';
    const iso = new Date().toISOString();
    const stamp = iso.replace(/[:.]/g, '-').replace('T', '-').replace('Z', '');
    return `${slug}_auto_${stamp}.json`;
  }

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
            pages = Array.from(pgSet).sort((a:number,b:number)=>a-b).join(', ');
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
    this.inlineExtractionPreferred.set(true);
    this.file.set(null); this.fileName.set(''); this.pdfUrl.set('');
    this.configEditorText = ''; this.isConfigValid.set(false); this.configErrors.set([]); this.validationMsg.set('');
    this.result.set(null); this.tiles.set([]);
    this.autoConfigBusy.set(false); this.autoConfigMsg.set(''); this.autoConfigError.set(''); this.autoConfigMaxPages.set(null);
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
    this.formItems.set(arr);
    this.onFormChange();
  }

  removeField(idx: number) { const arr = [...this.formItems()]; arr.splice(idx, 1); this.formItems.set(arr); this.onFormChange(); }
  onFormChange() { this.configEditorText = JSON.stringify(this.formItems(), null, 2); this.isConfigValid.set(false); this.validationMsg.set('Edited. Validate to check.'); this.inlineExtractionPreferred.set(true); }
  onTypeChange(i: number, newType: FormItem['type']) { this.formItems()[i].type = newType; if (newType === 'table') this.formItems()[i].multiline_value = true; this.onFormChange(); }
  onFormModeChange(evt: Event) {
    const checked = (evt.target as HTMLInputElement).checked;
    this.formMode.set(checked);
    this.inlineExtractionPreferred.set(true);
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
    const v = this.navVersion();
    return p ? `${base}#page=${p}&v=${v}` : `${base}#v=${v}`;
  }
  onTileClick(i: number) {
    const items = Array.isArray(this.result()) ? this.result() : [];
    const item = items[i];
    this.selectedIndex.set(i);
    const page = this.firstPageOf(item);
    if (page) this.gotoPage(page);
  }
  onTileAreaClick(i: number) {
    // First click expands, second click drills into entries if multi-entry; otherwise go to details
    const items = Array.isArray(this.result()) ? this.result() : [];
    const item = items[i];
    const expanded = this.expandedTileIndex();
    if (expanded !== i) {
      this.expandedTileIndex.set(i);
      this.drilledParentIndex.set(null);
      return;
    }
    if (this.isMultiEntry(item)) {
      this.drilledParentIndex.set(i);
      return;
    }
    // fallback to detail view
    this.onTileClick(i);
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

  pagesList(item: any): number[] {
    if (!item) return [];
    const t = this.guessType(item);
    if (t === 'key-value' && typeof item.page_number === 'number') return [Number(item.page_number)];
    if (t === 'summary' && Array.isArray(item.page_range) && item.page_range.length) {
      const a = Number(item.page_range[0]);
      const b = Number(item.page_range[1] ?? item.page_range[0]);
      const out = [a];
      if (!isNaN(b) && b !== a) out.push(b);
      return out.filter(n => !isNaN(n));
    }
    if ((t === 'bullet-points' || t === 'checkbox') && Array.isArray(item.value)) {
      const set = new Set<number>();
      for (const v of item.value) if (typeof v?.page_number === 'number') set.add(Number(v.page_number));
      return Array.from(set).sort((x:number,y:number)=>x-y);
    }
    if (t === 'table' && Array.isArray(item.page_numbers)) {
      return item.page_numbers.map((n:any)=>Number(n)).filter((n:number)=>!isNaN(n)).sort((a:number,b:number)=>a-b);
    }
    return [];
  }
  pagesForIndex(i: number): number[] {
    const arr = Array.isArray(this.result()) ? this.result() : [];
    return this.pagesList(arr[i]);
  }
  pagesForSelected(): number[] { return this.pagesList(this.currentItem()); }
  gotoPage(p: number, ev?: Event) {
    if (ev) ev.stopPropagation();
    if (typeof p === 'number' && !isNaN(p)) {
      this.pdfPage.set(p);
      this.navVersion.set(this.navVersion() + 1);
      // Force iframe re-creation so viewers that ignore hash updates still navigate
      this.pdfIframeVisible.set(false);
      setTimeout(() => this.pdfIframeVisible.set(true), 25);
    }
  }

  // Tile previews and drill-down helpers
  private clamp(text: string, maxChars: number): string {
    if (!text) return '';
    if (text.length <= maxChars) return text;
    return text.slice(0, maxChars).trimEnd() + '…';
  }
  isMultiEntry(item: any): boolean {
    const t = this.guessType(item);
    if (t === 'bullet-points' || t === 'checkbox') return Array.isArray(item?.value) && item.value.length > 1;
    if (t === 'table') return Array.isArray(item?.value) && item.value.length > 1;
    if (t === 'key-value') return Array.isArray(item?.multipage_detail) && item.multipage_detail.length > 1;
    return false;
  }
  tilePreviewText(item: any): string {
    const t = this.guessType(item);
    if (t === 'key-value' || t === 'summary') return this.clamp(String(item?.value ?? ''), 140);
    if (t === 'bullet-points' || t === 'checkbox') {
      const arr = Array.isArray(item?.value) ? item.value : [];
      const vals = arr.map((p:any)=>String(p?.value||'')).filter(Boolean);
      const shown = vals.slice(0, 2).join('\n• ');
      const more = vals.length > 2 ? `\n… (+${vals.length-2} more)` : '';
      return (shown ? '• ' : '') + shown + more;
    }
    if (t === 'table') {
      const rows = Array.isArray(item?.value) ? item.value.length : 0;
      const cols = Array.isArray(item?.columns) ? item.columns.length : 0;
      return `${rows} rows${cols?`, ${cols} cols`:''}`;
    }
    return this.clamp(typeof item?.value === 'string' ? item.value : JSON.stringify(item?.value ?? ''), 140);
  }
  tileFullText(item: any): string {
    const t = this.guessType(item);
    if (t === 'key-value' || t === 'summary') return String(item?.value ?? '');
    if (t === 'bullet-points' || t === 'checkbox') {
      const arr = Array.isArray(item?.value) ? item.value : [];
      const vals = arr.map((p:any)=>String(p?.value||'')).filter(Boolean);
      return (vals.length ? '• ' : '') + vals.join('\n• ');
    }
    if (t === 'table') {
      const rows = Array.isArray(item?.value) ? item.value.length : 0;
      const cols = Array.isArray(item?.columns) ? item.columns.length : 0;
      return `${rows} rows${cols?`, ${cols} cols`:''}`;
    }
    return typeof item?.value === 'string' ? item.value : JSON.stringify(item?.value ?? '');
  }
  subEntriesFor(item: any): Array<{label: string, page?: number}> {
    const t = this.guessType(item);
    const entries: Array<{label:string,page?:number}> = [];
    if (t === 'bullet-points' || t === 'checkbox') {
      const arr = Array.isArray(item?.value) ? item.value : [];
      for (const p of arr) entries.push({ label: String(p?.value||''), page: typeof p?.page_number==='number'?p.page_number:undefined });
    } else if (t === 'key-value' && Array.isArray(item?.multipage_detail)) {
      for (const f of item.multipage_detail) entries.push({ label: String(f?.value||''), page: typeof f?.page_number==='number'?f.page_number:undefined });
    } else if (t === 'table' && Array.isArray(item?.value)) {
      for (const row of item.value) {
        const cells = Array.isArray(row?.cells) ? row.cells : [];
        const sorted = cells.slice().sort((a:any,b:any)=>Number(a?.col||0)-Number(b?.col||0));
        const label = sorted.map((c:any)=>String(c?.value ?? '')).join(' | ');
        entries.push({ label, page: typeof row?.page_number==='number'?row.page_number:undefined });
      }
    }
    return entries;
  }
  drilledEntries(): Array<{label:string,page?:number}> {
    const idx = this.drilledParentIndex();
    const items = Array.isArray(this.result()) ? this.result() : [];
    const it = (idx===null)? null : items[idx];
    return it ? this.subEntriesFor(it) : [];
  }
  exitDrillDown() { this.drilledParentIndex.set(null); }
  getItemAt(i: number | null): any {
    if (i===null) return null;
    const arr = Array.isArray(this.result()) ? this.result() : [];
    return arr[i];
  }
  onPdfLockToggle(evt: Event) {
    const input = evt.target as HTMLInputElement;
    this.pdfLocked.set(!!input?.checked);
  }

   // Table helpers for schema-aware rendering and drill-down
  private getTableColumns(it: any): string[] {
    if (!it) return [];
    const cols: string[] = Array.isArray(it.columns) && it.columns.length ? it.columns.slice() : [];
    if (cols.length) return cols;
    const rows = Array.isArray(it.value) ? it.value.slice(0, 50) : [];
    const indexSet = new Set<number>();
    const nameByIndex = new Map<number, string>();

    // If the first logical row looks like a header (row===1 and has string labels), prefer it
    const headerRow = rows.find((r:any) => r && (r.row === 1 || r.row === '1'));
    if (headerRow && Array.isArray(headerRow.cells) && headerRow.cells.length) {
      let hasText = false;
      for (const c of headerRow.cells) {
        const idx = Number(c?.col);
        const val = c?.value;
        if (!isNaN(idx)) {
          indexSet.add(idx);
          if (val !== undefined && val !== null && String(val).trim() !== '') {
            nameByIndex.set(idx, String(val));
            if (String(val).trim().length > 0) hasText = true;
          }
        }
      }
      // If headerRow appears valid, return names in index order
      if (hasText) {
        const ordered = Array.from(indexSet).sort((a:number,b:number)=>a-b);
        return ordered.map(i => nameByIndex.get(i) || `col_${i}`);
      }
    }

    // Fallback: infer columns from any row cells by index and optional col_name
    for (const row of rows) {
      const cells = Array.isArray(row?.cells) ? row.cells : [];
      for (const c of cells) {
        const idx = Number(c?.col);
        if (!isNaN(idx)) {
          indexSet.add(idx);
          if (c?.col_name && !nameByIndex.has(idx)) nameByIndex.set(idx, String(c.col_name));
        }
      }
    }
    const ordered = Array.from(indexSet).sort((a:number,b:number)=>a-b);
    return ordered.map(i => nameByIndex.get(i) || `col_${i}`);
  }
  get tableHeaders(): string[] { return [...this.getTableColumns(this.currentItem()), 'Page']; }
  private rowToValues(it: any, row: any, headers: string[]): string[] {
     const cells = Array.isArray(row?.cells) ? row.cells : [];
     const byColIndex = new Map<number, string>();
     const byColName = new Map<string, string>();
     for (const c of cells) {
       const idx = Number(c?.col);
       if (!isNaN(idx)) byColIndex.set(idx, String(c?.value ?? ''));
       const nm = c?.col_name; if (nm) byColName.set(String(nm), String(c?.value ?? ''));
     }
     const values: string[] = [];
    headers.forEach((h, i) => {
      if (h === 'Page') {
        // Page is not a cell; take it from row.page_number
        values.push(row && row.page_number !== undefined && row.page_number !== null ? String(row.page_number) : '');
        return;
      }
      let v = byColName.get(h);
      if (v === undefined) v = byColIndex.get(i+1);
      values.push(v ?? '');
    });
     return values;
   }
  get tablePreviewRows(): string[][] {
    const it = this.currentItem();
    const headers = this.tableHeaders; // include the 'Page' column
    const rows = Array.isArray(it?.value) ? it.value.slice(0, 10) : [];
    return rows.map((r:any)=>this.rowToValues(it, r, headers));
  }

  // Return all table rows (as arrays of cell strings) for the current item
  getAllTableRows(): string[][] {
    const it = this.currentItem(); if (!it) return [];
    const headers = this.tableHeaders;
    const rows = Array.isArray(it?.value) ? it.value : [];
    return rows.map((r:any)=>this.rowToValues(it, r, headers));
  }

  totalTablePages(): number {
    const all = this.getAllTableRows();
    const size = Math.max(1, Number(this.tablePageSize()));
    return Math.max(1, Math.ceil(all.length / size));
  }

  paginatedTableRows(): string[][] {
    const all = this.getAllTableRows();
    const size = Math.max(1, Number(this.tablePageSize()));
    const idx = Math.max(0, Number(this.tablePageIndex()));
    const start = idx * size;
    return all.slice(start, start + size);
  }

  prevTablePage() {
    const idx = this.tablePageIndex();
    if (idx > 0) this.tablePageIndex.set(idx - 1);
  }
  nextTablePage() {
    const idx = this.tablePageIndex();
    const last = this.totalTablePages() - 1;
    if (idx < last) this.tablePageIndex.set(idx + 1);
  }
  onTablePageSizeChange(size: number) {
    this.tablePageSize.set(Number(size));
    this.tablePageIndex.set(0); // reset to first page
  }

  // Export all table rows as CSV (Excel-friendly). Use full table rather than only current page.
  exportTableCSV() {
    const it = this.currentItem(); if (!it) return;
    const headers = this.tableHeaders;
    const rows = this.getAllTableRows();
    if (!rows.length) return;
    const csvRows: string[] = [];
    csvRows.push(headers.map((h: any) => '"' + String(h).replace(/"/g, '""') + '"').join(','));
    for (const r of rows) csvRows.push((r as any[]).map((c: any) => '"' + String(c ?? '').replace(/"/g, '""') + '"').join(','));
    const blob = new Blob([csvRows.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = ((it.field_name||'table').replace(/[^a-z0-9_\-]/gi,'_')) + '.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  // Load a demo JSON result from frontend assets (useful when backend isn't available)
  loadLocalOutput() {
    this.api.getLocalOutput('dummy_statement.json').subscribe({
      next: (r: any) => {
        const payload = (r && Array.isArray(r)) ? r : (r && r.result) ? r.result : r;
        console.log('loadLocalOutput: payload length=', Array.isArray(payload)?payload.length:0, payload);
        this.onExtractionDone(payload);
      },
      error: (e: any) => this.onExtractionError(e)
    });
  }

}
