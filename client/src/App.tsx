import './App.css'
import Layout from './base/Layout'
import { Route, Routes } from 'react-router-dom'
import { Paths } from './Paths'

function App() {

  return (
    <Routes>
        <Route path="/" element={<Layout />}>
          {Paths.map((item) => (
            <Route key={item.path} path={item.path} element={item.component} />
          ))}
        </Route>
    </Routes>
  )
}

export default App
