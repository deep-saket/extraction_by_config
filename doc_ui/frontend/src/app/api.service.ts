import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  base = 'http://localhost:8001';

  constructor(private http: HttpClient) {}

  health(): Observable<any> {
    return this.http.get(`${this.base}/health`);
  }

  schemaVersion(): Observable<{ schema_hash: string }> {
    return this.http.get<{ schema_hash: string }>(`${this.base}/schemas/version`);
  }

  listConfigs(): Observable<{ configs: string[] }> {
    return this.http.get<{ configs: string[] }>(`${this.base}/configs/list`);
  }

  getConfig(name: string): Observable<any> {
    return this.http.get(`${this.base}/configs/get`, { params: { name } });
  }

  validateConfig(content: any): Observable<{ ok: boolean; errors?: any }>{
    return this.http.post<{ ok: boolean; errors?: any }>(`${this.base}/configs/validate`, content);
  }

  saveConfig(name: string, content: any): Observable<{ ok: boolean; name?: string; errors?: any }>{
    return this.http.post<{ ok: boolean; name?: string; errors?: any }>(`${this.base}/configs/save`, { name, content });
  }

  performDE(file: File, configName: string): Observable<any> {
    const form = new FormData();
    form.append('pdf', file);
    form.append('config_name', configName);
    return this.http.post(`${this.base}/perform_de`, form);
  }

  performDEWithInlineConfig(file: File, configJson: any): Observable<any> {
    const form = new FormData();
    form.append('pdf', file);
    form.append('config_json', JSON.stringify(configJson));
    return this.http.post(`${this.base}/perform_de`, form);
  }

  // Added helper to load a local JSON placed in the frontend `assets/` folder (useful for testing)
  getLocalOutput(name: string): Observable<any> {
    return this.http.get(`/assets/${name}`);
  }

  generateAutoConfig(file: File, maxPages?: number | null): Observable<{ items: any[]; count: number }> {
    const form = new FormData();
    form.append('pdf', file);
    if (typeof maxPages === 'number' && !isNaN(maxPages)) {
      form.append('max_pages', String(maxPages));
    }
    return this.http.post<{ items: any[]; count: number }>(`${this.base}/auto_config/generate`, form);
  }

}
