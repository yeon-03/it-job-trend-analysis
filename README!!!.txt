폴더 디렉터리에서 cmd실행후 다음 입력

uvicorn server:app --reload --port 8000

그 뒤 chrome, edge같은 웹사이트 주소에 다음을 입력

http://localhost:8000/

서버에 return되는 값을 보고싶으면 다음 주소 
	
http://localhost:8000/docs

