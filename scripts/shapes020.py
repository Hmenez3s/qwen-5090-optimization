from pathlib import Path
import json
for name in ['Qwen3.8-27B-NVFP4-LMHead4','DFlash2-NVFP4-RTNcal']:
 d=json.loads((Path('/home/operator/qwen38-5090/models')/name/'config.json').read_text());t=d.get('text_config',d);print(name,{k:t.get(k) for k in ['model_type','hidden_size','intermediate_size','num_hidden_layers','vocab_size']})
