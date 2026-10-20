import sys
import os
import re
import json
import urllib.request
import subprocess

def run_update(url):
    print(f"[*] Starting update pipeline for URL: {url}")
    
    # 1. Fetch WeChat Article
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
    
    title_m = re.search(r'<meta property="og:title" content="(.*?)"', html)
    voice_m = re.search(r'voice_encode_fileid="([^"]+)"', html)
    raw_title = title_m.group(1) if title_m else "考研政治带背"
    
    if not voice_m:
        print("[!] No voice_encode_fileid found in article!")
        return
        
    voice_id = voice_m.group(1)
    audio_url = f"https://res.wx.qq.com/voice/getvoice?mediaid={voice_id}"
    print(f"[+] Found Article Title: {raw_title}")
    print(f"[+] Found Voice ID: {voice_id}")
    print(f"[+] Audio URL: {audio_url}")
    
    # Identify Speaker & Series
    is_tuijie = bool(re.search(r'腿姐|陆寓丰', raw_title + html[:3000]))
    day_num_m = re.search(r'D(\d+)|DAY\s*(\d+)', raw_title, re.I)
    num = int(day_num_m.group(1) or day_num_m.group(2)) if day_num_m else 1

    if is_tuijie:
        series = "tuijie"
        speaker = "腿姐 (陆寓丰)"
        badge = "腿姐冲刺"
        day_key = f"tj_d{num:02d}"
        day_label = f"腿姐 D{num:02d}"
        book_sub = "《冲刺背诵手册》下册"
    else:
        series = "hanxue"
        speaker = "韩雪老师"
        badge = "肖秀荣精读"
        day_key = f"d{num:02d}"
        day_label = f"韩雪 D{num:02d}"
        book_sub = "《背诵手册》上册"
        
    print(f"[+] Target Series: {series} ({speaker}), Key: {day_key} ({day_label})")
    
    # Download audio backup
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_file_name = f"{'tuijie' if is_tuijie else 'hanxue'}_D{num}.mp3"
    audio_backup_path = os.path.join(audio_dir, audio_file_name)
    try:
        print(f"[*] Downloading audio backup to {audio_backup_path}...")
        with urllib.request.urlopen(urllib.request.Request(audio_url, headers=headers)) as resp, open(audio_backup_path, 'wb') as f:
            f.write(resp.read())
        print(f"[+] Audio downloaded ({os.path.getsize(audio_backup_path)} bytes)")
    except Exception as e:
        print(f"[!] Warning: Audio download backup failed: {e}")

    # 2. Call DashScope ASR
    import dashscope
    from dashscope.audio.asr import Transcription
    dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
    
    print("[*] Calling DashScope Paraformer-v2 ASR...")
    res = Transcription.async_call(model='paraformer-v2', file_urls=[audio_url])
    if res.status_code != 200:
        print(f"[!] ASR submission failed: {res}")
        return
        
    task_id = res.output['task_id']
    print(f"[*] ASR Task submitted: {task_id}, waiting for results...")
    st_res = Transcription.wait(task=task_id)
    if st_res.status_code != 200:
        print(f"[!] ASR execution failed: {st_res}")
        return
        
    def find_url(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'transcription_url' and v: return v
                r = find_url(v)
                if r: return r
        elif isinstance(obj, list):
            for item in obj:
                r = find_url(item)
                if r: return r
        return None
        
    t_url = find_url(st_res.output)
    if not t_url:
        print("[!] No transcription URL found!")
        return
        
    print(f"[+] Downloading ASR transcripts...")
    asr_data = json.loads(urllib.request.urlopen(t_url).read().decode('utf-8'))
    sents = asr_data['transcripts'][0]['sentences']
    print(f"[+] Transcribed {len(sents)} sentences.")
    
    # 3. Semantic Paragraph Clustering
    paras = []
    current_para = []
    lead_keywords = [
        ("【开篇导语】", r"同学们大家好|欢迎回到|今天是第|早上好|小太阳|聪明勇敢"),
        ("【核心概念】", r"到底什么是|何为|概念|范畴|定义|第一个纲领|党章"),
        ("【通俗举例】", r"比如|比如说|举个例子|就像|四川担担面|川全"),
        ("【考点点拨】", r"标注一下|画起来|重点是|注意一下|一定要注意|打两颗星|周年纪念|须知政权"),
        ("【维度剖析】", r"第一方面|第二方面|第三方面|第一点|第二点|第三点|首先|其次|再次|第一|第二|第三|第四|第五|第六"),
        ("【深挖辨析】", r"深挖|为什么|怎么理解|关系是什么|对立|统一|基本一致|完全相同"),
        ("【易错排雷】", r"错误|犯错误|形而上学|唯心主义|经典干扰|干扰项|陷阱|左倾|右倾|关门主义"),
        ("【命题分析】", r"命题分析|历年考查|真题考查|真题出题|频次|单选|多选"),
        ("【复盘结语】", r"总结一下|最后|一言以蔽之|明天见|今天的内容|加油")
    ]
    
    for s in sents:
        txt = s['text']
        should_break = False
        if len(current_para) >= 2:
            for _, pat in lead_keywords:
                if re.search(pat, txt):
                    should_break = True
                    break
        if len(current_para) >= 6:
            should_break = True
            
        if should_break and current_para:
            pst = round(current_para[0]['begin_time'] / 1000.0, 2)
            pet = round(current_para[-1]['end_time'] / 1000.0, 2)
            p_lead = "📌 重点讲授："
            for lead_name, pat in lead_keywords:
                if re.search(pat, current_para[0]['text']):
                    p_lead = lead_name + " "
                    break
            paras.append({
                "lead": p_lead,
                "st": pst,
                "et": pet,
                "sentences": [
                    {
                        "st": round(x['begin_time'] / 1000.0, 2),
                        "et": round(x['end_time'] / 1000.0, 2),
                        "text": x['text']
                    } for x in current_para
                ]
            })
            current_para = []
        current_para.append(s)
        
    if current_para:
        pst = round(current_para[0]['begin_time'] / 1000.0, 2)
        pet = round(current_para[-1]['end_time'] / 1000.0, 2)
        paras.append({
            "lead": "📝 今日复盘：",
            "st": pst,
            "et": pet,
            "sentences": [
                {
                    "st": round(x['begin_time'] / 1000.0, 2),
                    "et": round(x['end_time'] / 1000.0, 2),
                    "text": x['text']
                } for x in current_para
            ]
        })
        
    cards = []
    card_size = 4 if is_tuijie else 5
    for c_i in range(0, len(paras), card_size):
        sub_paras = paras[c_i : c_i + card_size]
        cards.append({
            "id": f"{day_key}_c{len(cards)+1}",
            "tag": f"模块 {len(cards)+1}",
            "title": f"{sub_paras[0]['lead'].strip()} · 语段 {c_i+1}-{min(c_i+card_size, len(paras))}",
            "sub": f"涵盖 {len(sub_paras)} 个重点小节",
            "start": sub_paras[0]['st'],
            "end": sub_paras[-1]['et'],
            "paras": sub_paras
        })
        
    # Clean up topic title
    topic_clean = re.sub(r'【.*?】|‼️|📢', '', raw_title).strip()
    day_data = {
        "id": day_key,
        "day": day_label,
        "series": series,
        "speaker": speaker,
        "title": f"{day_label} · {topic_clean}",
        "sub": f"{book_sub} 每日带背",
        "audioUrl": audio_url,
        "duration": sents[-1]['end_time'] / 1000.0,
        "cards": cards
    }
    
    # 4. Save to data/{day_key}.json
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f"{day_key}.json"), "w", encoding="utf-8") as f:
        json.dump(day_data, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {day_key}.json")
    
    # 5. Update manifest.json
    manifest_path = os.path.join(data_dir, "manifest.json")
    m_list = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            m_list = json.load(f)
            
    dur_m = int((sents[-1]['end_time'] / 1000.0) // 60)
    dur_s = int((sents[-1]['end_time'] / 1000.0) % 60)
    new_entry = {
        "id": day_key,
        "day": day_label,
        "series": series,
        "speaker": speaker,
        "badge": badge,
        "title": topic_clean,
        "pages": book_sub,
        "desc": f"涵盖 {len(paras)} 个考点语段精讲",
        "duration": f"{dur_m:02d}:{dur_s:02d}",
        "file": f"data/{day_key}.json"
    }
    m_list = [x for x in m_list if x["id"] != day_key]
    m_list.append(new_entry)
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m_list, f, ensure_ascii=False, indent=2)
    print(f"[+] Updated manifest.json (Total {len(m_list)} days)")
    
    # 6. Update database.js
    days_dict = {}
    for item in m_list:
        d_path = os.path.join(data_dir, f"{item['id']}.json")
        if os.path.exists(d_path):
            with open(d_path, "r", encoding="utf-8") as f:
                days_dict[item["id"]] = json.load(f)
                
    db_content = f"window.RECITE_DATABASE = {json.dumps({'manifest': m_list, 'days': days_dict}, ensure_ascii=False, indent=2)};\n"
    with open(os.path.join(data_dir, "database.js"), "w", encoding="utf-8") as f:
        f.write(db_content)
    print(f"[+] Updated database.js")
    
    # 7. Git commit & push
    repo_dir = os.path.dirname(__file__)
    print("[*] Pushing to GitHub...")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", f"feat: update {day_label} - {topic_clean}"], cwd=repo_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, check=True)
    print(f"[✓] Deployment complete! Live at: https://ylhyc.github.io/politics-recite/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_update(sys.argv[1])
    else:
        print("Usage: python update_day.py <wechat_article_url>")
