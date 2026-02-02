# React Native Quick Reference (for Interview)

## Most Important Concepts

### 1. Components = Views
```javascript
<View>          // Like ViewGroup
  <Text>Hello</Text>   // Like TextView
  <Image />     // Like ImageView
</View>
```

### 2. State = LiveData
```javascript
const [name, setName] = useState('');
// Like: val name = MutableLiveData<String>()
```

### 3. Props = Bundle Arguments
```javascript
<MyComponent title="Hello" />
// Like: Bundle.putString("title", "Hello")
```

### 4. Hooks = Lifecycle
```javascript
useEffect(() => {
  // onCreate
  return () => {
    // onDestroy
  };
}, []);
```

## Interview Answers

**Q: "You know Android. Why React Native?"**

**A:** "To build for iOS and Android simultaneously. Faster iteration, shared codebase. But my Android experience is valuable - I know when to use native modules for performance or platform-specific features."

**Q: "How fast can you learn React Native?"**

**A:** "The concepts transfer directly. I learned ML in 24 days and built 19 projects. React Native is easier - it's just mobile concepts I already know in different notation. I've already studied the architecture and built component designs."

## My Edge

**Not a beginner learning mobile development.**
**A mobile expert learning new syntax.**

HUGE difference! 🚀
