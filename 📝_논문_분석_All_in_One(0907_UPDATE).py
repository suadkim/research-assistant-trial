import streamlit as st
import re
import bs4
import os
from fake_useragent import UserAgent
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
from openai import OpenAI
import anthropic
from elevenlabs import generate, play, set_api_key
from pydub import AudioSegment
import io
import requests
import time

st.title("논문 요약 및 팟캐스트 생성")

elevenlabs_api_key = "자신의 API KEY"
openai_api_key = "자신의 API KEY"
anthropic_api_key = "자신의 API KEY"
gemini_api_key = "자신의 API KEY"

if elevenlabs_api_key and openai_api_key and anthropic_api_key and gemini_api_key:
   set_api_key(elevenlabs_api_key)

   # ArXiv paper URL input
   user_input_url = st.text_input("* ArXiv 논문 링크를 입력하세요: ")

   if user_input_url:
       def convert_arxiv_url(user_input_url):
           # arxiv ID를 추출하는 정규 표현식 패턴
           arxiv_id_pattern = r'(\d{4}\.\d{5})(v\d+)?'

           # 새로운 변환 로직 시도
           if user_input_url.startswith(("https://arxiv.org/abs/", "https://www.arxiv.org/abs/")):
               match = re.search(arxiv_id_pattern, user_input_url)
               if match:
                   arxiv_id = match.group(1)
                   converted_url = f"https://arxiv.org/html/{arxiv_id}"

                   # 변환된 URL로 문서 로드 시도
                   loader = WebBaseLoader(converted_url, header_template={
                       'User-Agent': UserAgent().chrome,
                       })
                   docs = loader.load()

                   # 타이틀 확인
                   if "arXiv e-print repository" in docs[0].metadata['title']:
                       # 타이틀에 "arXiv e-print repository"가 포함되면 기존 로직으로 넘어감
                       converted_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

                   return converted_url
           elif user_input_url.startswith(("https://arxiv.org/pdf/", "https://www.arxiv.org/pdf/")):
               match = re.search(arxiv_id_pattern, user_input_url)
               if match:
                   arxiv_id = match.group(1)
                   converted_url = f"https://arxiv.org/html/{arxiv_id}"

                   # 변환된 URL로 문서 로드 시도
                   loader = WebBaseLoader(converted_url, header_template={
                       'User-Agent': UserAgent().chrome,
                       })
                   docs = loader.load()

                   # 타이틀 확인
                   if "arXiv e-print repository" in docs[0].metadata['title']:
                       # 타이틀에 "arXiv e-print repository"가 포함되면 기존 로직으로 넘어감
                       converted_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

                   return converted_url
           elif user_input_url.startswith(("https://arxiv.org/html/", "https://www.arxiv.org/html/")):
               return user_input_url

           # 실패한 경우
           st.error("지원되지 않는 URL 형식입니다.")
           return None

       converted_url = convert_arxiv_url(user_input_url)

       if converted_url:
           loader = WebBaseLoader(converted_url, header_template={
             'User-Agent': UserAgent().chrome,
               })

           docs = loader.load()

           original_text = docs[0].page_content
           # 불필요한 공백, 줄바꿈, 탭 제거
           cleaned_text = re.sub(r'\s+', ' ', original_text).strip()

           title = docs[0].metadata['title']
           source = docs[0].metadata['source']

           st.write(f"* 제목: {title}")
           st.write(f"* 출처: {source}")

           # 논문의 References 부분 찾는 함수
           def find_references_index(text):
               return text.find("References")

           index = find_references_index(cleaned_text)

           # 논문 처음부터 References 직전까지 부분 추출
           paper_content_before_references = cleaned_text[:index]
# Gemini 모델 설정
           gemini_model_name = "gemini-1.5-pro-exp-0801"
           gemini_temperature = 0.4
           gemini_top_p = 0.95
           gemini_top_k = 64
           gemini_max_output_tokens = 8192
           gemini_response_mime_type = "text/plain"
           gemini_threshold = "BLOCK_NONE"

           # gemini 모델 설정
           genai.configure(api_key=gemini_api_key)

           generation_config = {
             "temperature": gemini_temperature,
             "top_p": gemini_top_p,
             "top_k": gemini_top_k,
             "max_output_tokens": gemini_max_output_tokens,
             "response_mime_type": gemini_response_mime_type,
           }

           safety_settings = [
             {
               "category": "HARM_CATEGORY_HARASSMENT",
               "threshold": gemini_threshold
             },
             {
               "category": "HARM_CATEGORY_HATE_SPEECH",
               "threshold": gemini_threshold
             },
             {
               "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
               "threshold": gemini_threshold
             },
             {
               "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
               "threshold": gemini_threshold
             }
           ]

           # Gemini 모델에 적용하는 system instruction
           section_summary_instruction = """
           다음 글은 논문 전체(또는 일부)를 그대로 복사한 거야. 논문은 구조적인 글로 잘 알려져있는데, 그 점을 감안하여 섹션별로(e.g. Introduction, Method, ...) 3 ~ 5 문장의 분량으로 요약해.
           글에 포함되어 있는 섹션은 하나도 빠짐없이 들어가야함에 주의해.
           또한, 각각의 요약 내용에는 그 섹션의 핵심이 꼭 포함되어야 함에 주의해. 요약 언어는 *영어*를 사용할 것. \n\n ------
           """

           gemini_model = genai.GenerativeModel(model_name=gemini_model_name,
                                         generation_config=generation_config,
                                         safety_settings=safety_settings,
                                         system_instruction=section_summary_instruction
                                         )

           # 섹션별 요약 (영어)
           prompt_with_system_instruction = section_summary_instruction + paper_content_before_references
           tokens_of_prompt_with_instruction = int(str(gemini_model.count_tokens(prompt_with_system_instruction)).replace("total_tokens: ",""))
           chunk_size = 24000
           num_chunks = (tokens_of_prompt_with_instruction // chunk_size) + 1

           with st.spinner('섹션별 요약 (영어) 생성 중...'):
               section_summarization = ""
   
               for i in range(num_chunks):
                   start = i * chunk_size
                   end = min((i + 1) * chunk_size, tokens_of_prompt_with_instruction)
   
                   chunk = paper_content_before_references[start:end]
                   section_summarization_chunk = gemini_model.generate_content(chunk).text
   
                   section_summarization += section_summarization_chunk
   
                   if i < num_chunks - 1:
                       time.sleep(60)

           st.markdown("## 섹션별 요약 (영어):")
           st.write(section_summarization)
# OpenAI(ChatGPT) 모델 설정
           openai_model = OpenAI(api_key=openai_api_key)

           openai_model_name = "gpt-4o"
           temperature = 0.7
           max_tokens = 1000

           # OpenAI 모델에 적용하는 system instruction 모음
           section_tag_instruction = "주어진 요약문에서 논문의 섹션명만 리스트로 뽑아서 그것만 출력할 것. 넘버링은 포함시키지말고 텍스트만 추출할 것."

           trim_instruction = "특수 기호 등은 제외하고, 텍스트만 깔끔하게 줄글로 이어지도록 재작성해줘. 또한, 고유명사 등은 영어 그대로 사용하고, 그 외에는 모두 한국어로 자연스럽게 번역할 것."

           mermaid_system_instruction = "너는 mermaid 코드 전문가야."

           with st.spinner('줄글 요약 (한국어) 생성 중...'):
               # 줄글 요약 (한국어)
               trimmed_summarization = openai_model.chat.completions.create(
                 model="gpt-4o",
                 messages=[
                   {"role": "system", "content": trim_instruction},
                   {"role": "user", "content": section_summarization},
                 ]
               ).choices[0].message.content

           st.markdown("## 줄글 요약 (한국어):")
           st.write(trimmed_summarization)
           
           with st.spinner('마인드맵 요약 생성 중...'):            
               # 마인드맵 요약 (영어)
               tag_list = openai_model.chat.completions.create(
                 model="gpt-4o",
                 messages=[
                   {"role": "system", "content": section_tag_instruction},
                   {"role": "user", "content": section_summarization},
                 ]
               ).choices[0].message.content
   
               # mermaid code 만들 때 사용하는 user prompt
               mermaid_converter_instruction = f"""
               * Mermaid Code Format Example*
               ```
               mindmap
                 root((Title of the paper))
                   Origins
                     Long history
                     ::icon(fa fa-book)
                     Popularisation
                       British popular psychology author Tony Buzan
                   Research
                     On effectivness<br/>and features
                     On Automatic creation
                       Uses
                           Creative techniques
                           Strategic planning
                           Argument mapping
                   Tools
                     Pen and paper
                     Mermaid
               ```
   
               위의 Meramid Code를 참고하여 다음 요약된 논문 내용을 Mermaid Code의 mindmap으로 만들어줘.
               이때, mindmap의 구조는 꼭 다음 형식을 대원칙으로 지키도록 명심해.
   
               ```
               1. root가 {title}가 되어야 한다.
               2. mindmap의 첫번째 가지는 섹션명이 되어야한다. 이때 섹션명은 다음을 참고할 것. \n\n {tag_list} \n\n
               3. mindmap의 두번째 가지는 다음 섹션별 요약을 참고하여 각 섹션의 가지로서, 키워드 중심으로 관련성을 잘 파악하여 곁가지를 칠 것. \n{section_summarization}
               4. mermaid 코드에서는 가지 안에 들어가는 내용에 괄호 등의 특수 기호 혹은 숫자가 !!절대로 포함되어서는 안 된다!!
               ```
               !!그 외의 말은 절대 출력하지마.!!
               """
   
               mermaid_code = openai_model.chat.completions.create(
                  model="gpt-4o",
                  messages=[
                    {"role": "system", "content": mermaid_system_instruction},
                    {"role": "user", "content": mermaid_converter_instruction},
                  ]
                ).choices[0].message.content
    
               start_index = mermaid_code.find("mindmap")
               end_index = mermaid_code.rfind("\n")
               mermaid_code = mermaid_code[start_index:end_index].strip()
    
               BASE_URL = "https://diagrams.helpful.dev"
    
               def post_render_diagram(title: str, diagramSource: str):
                   data = {
                       "title": title,
                       "diagramSource": diagramSource
                   }
                   response = requests.post(f"{BASE_URL}/v2/render-diagram", json=data)
                   return response.json()
    
               response = post_render_diagram(
                   title="Mindmap Summary",
                   diagramSource=mermaid_code
               )
    
               say_to_user = response.get("sayToUser", "")

               # URL 추출을 위한 정규 표현식 패턴
               image_pattern = r'!\[\]\((https://diagrams\.helpful\.dev/d/[^\s)]+)\)'
               fullscreen_pattern = r'\[View fullscreen\]\((https://diagrams\.helpful\.dev/d/[^\s)]+)\)'
               download_pattern = r'\[Download png\]\((https://diagrams\.helpful\.dev/d/[^\s)]+)\)'
               edit_code_pattern = r'\[Edit with code\]\((https://diagrams\.helpful\.dev/s/[^\s)]+)\)'
               edit_miro_pattern = r'\[Edit with Miro using drag and drop\]\((https://diagrams\.helpful\.dev/m/[^\s)]+)\)'

               image_match = re.search(image_pattern, say_to_user)
               fullscreen_match = re.search(fullscreen_pattern, say_to_user)
               download_match = re.search(download_pattern, say_to_user)
               edit_code_match = re.search(edit_code_pattern, say_to_user)
               edit_miro_match = re.search(edit_miro_pattern, say_to_user)

               if image_match:
                   image_url = image_match.group(1)
                   fullscreen_url = fullscreen_match.group(1) if fullscreen_match else ""
                   download_url = download_match.group(1) if download_match else ""
                   edit_code_url = edit_code_match.group(1) if edit_code_match else ""
                   edit_miro_url = edit_miro_match.group(1) if edit_miro_match else ""

                   st.markdown("## 마인드맵 요약 :")
                   st.markdown(f"![마인드맵 이미지]({image_url})")
                   st.markdown(f"[마인드맵 크게 보기]({fullscreen_url})")
                   st.markdown(f"[마인드맵 이미지 다운로드]({download_url})")
                   st.markdown(f"[마인드맵 수정 - 노코드 버전]({edit_miro_url})")
                   st.markdown(f"[마인드맵 수정 - 코드 버전]({edit_code_url})")
               else:
                   st.error("마인드맵 생성에 실패했습니다. 다시 시도해주세요.")

           # Anthropic(Claude) 모델 설정
           anthropic_model = anthropic.Anthropic(api_key=anthropic_api_key)

           anthropic_model_name="claude-3-5-sonnet-20240620"
           anthropic_max_tokens = 4096
           anthropic_temperature = 0.7

           if "podcast_text" not in st.session_state:
               with st.spinner('팟캐스트 대본 생성 중...'):
                  # 한국어 팟캐스트 대본 생성
                  podcast_instruction = """주어진 글은 생성형 인공지능이 논문을 요약한 거야. 이 요약문을 논문 팟캐스트 진행자가 읽는 글로 만들어줘. 문체가 정말 사람이 쓴 것 같이 자연스러워야함에 주의해.
                  또한, 너가 출력한 팟캐스트 형식의 글을 그대로 tts를 통해 인공지능이 읽게 할 거라서 다음 원칙을 따라서 대본을 생성해줘. 
                  ```
                  * 원칙 :
                  1. 숫자와 기호는 문맥에 따라서 발음이 나는대로 한글로 변환할 것. e.g.) 2.56% -> 이쩜오륙퍼센트
                  2. 영어는 발음이 나는대로 한글로 변환할 것. e.g.) Task -> 태스크, Method -> 메서드
                  3. 맨 앞에는 다음 인사말로 항상 먼저 시작할 것 : '안녕하세요, 여러분! 박준의 논문 팟캐스트입니다. 오늘 소개해드릴 논문은 ~~~'
                  4. 맨 뒤에는 다음과 같이 마무리할 것 : '그럼 오늘 논문 팟캐스트는 여기서 마치겠습니다. 들어주셔서 감사합니다. 박준이었습니다!'
                  ```
                  그리고, 수정한 글의 출력 외에는 아무 출력도 하지마."""
            
                  prompt_with_podcast_instruction = podcast_instruction + "\n\n-------" + trimmed_summarization
            
                  st.session_state.podcast_text = anthropic_model.messages.create(
                      model=anthropic_model_name,
                      max_tokens=anthropic_max_tokens,
                      temperature=anthropic_temperature,
                      system="너는 팟캐스트 대본 작가야.",
                      messages=[
                          {
                              "role": "user",
                              "content": [
                                  {
                                      "type": "text",
                                      "text": prompt_with_podcast_instruction
                                  }
                              ]
                          }
                      ]
                  ).content[0].text
        
           # 모델 응답 출력
           st.markdown("## 팟캐스트 대본 : ")
           st.write(st.session_state.podcast_text)            

           if "file_name" not in st.session_state:
               with st.spinner('오디오 파일 생성 중...'):
                   # Split the text into sentences using regular expression
                   sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', st.session_state.podcast_text)
            
                   # Chunk the sentences into groups of one
                   chunks = [sentence.strip() for sentence in sentences if sentence.strip()]
            
                   audios = []
                   voice = "Lily"
                   elevenlabs_model_name = "eleven_multilingual_v2"            
                   attempt = 0
                   while attempt < 3:
                       try:
                           for i in range(len(chunks)):
                               audio = generate(text=chunks[i], voice=voice, model=elevenlabs_model_name)
                               audios.append(audio)
                           st.success(f"{voice}의 목소리로 성공적으로 오디오를 생성하였습니다.")                            
                           break
                       except Exception as e:
                           st.error(f"{voice} 목소리로 실행 중 오류 발생 (시도 {attempt + 1}/3): {str(e)}")
                           attempt += 1
                           if attempt < 3:
                               time.sleep(1)  # 1초 대기 후 다시 시도
                           else:
                               st.error(f"{voice} 목소리로 3번 시도 후에도 오류 발생. 다시 시도해주세요.")
           
                   # 오디오 파일 생성
                   st.session_state.file_name = "podcast.mp3"
            
                   with open(st.session_state.file_name, "wb") as f:
                       for audio in audios:
                           f.write(audio)
        
               # 오디오 재생
               audio_file = open(st.session_state.file_name, 'rb')
               audio_bytes = audio_file.read()
               st.markdown(f"## 논문 팟캐스트 by {voice} : ")
               st.audio(audio_bytes, format='audio/mp3')
