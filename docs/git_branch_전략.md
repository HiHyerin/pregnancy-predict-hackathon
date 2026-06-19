## 개발 규칙

### Git 브랜치 전략

```
main
 ├─ feature/rag
 ├─ feature/login
 └─ feature/chatbot
```

**브랜치 명명 규칙**

```
feature/기능명
fix/버그명
hotfix/긴급수정
refactor/리팩토링명
```

---

### Commit 규칙

```
feat: 기능 추가
fix: 버그 수정
refactor: 코드 개선
docs: 문서 수정
test: 테스트 추가
chore: 기타 작업
```

---

### 코드 리뷰 규칙

- 최소 1명 이상 승인 후 Merge
- 리뷰는 코드가 아닌 문제를 지적한다.
- 리뷰 의견은 근거를 함께 작성한다.