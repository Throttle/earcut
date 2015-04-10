### earcut.py

earcut.py is a simple port of earcut.js by Vladimir Agafonkin ([https://github.com/mourner/earcut-js](https://github.com/mourner/earcut-js))

### Usage

```
import earcut
earcut(points, returnIndices)
```


`points`: A 2D array of points coordinates: `[[int/float, int/float]]`

`returnIndices (optional, False by default)`: Flag to get results in the form of flat index and vertex arrays by passing true as a second argument to earcut (convenient for uploading results directly to WebGL as buffers):.