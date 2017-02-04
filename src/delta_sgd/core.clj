(ns delta-sgd.core)

(def alpha 0.9)

(def samples [{:xs [0 0 1.0] :d 0} 
              {:xs [0 1.0 1.0] :d 0} 
              {:xs [1.0 0 1.0] :d 1.0}
              {:xs [1.0 1.0 1.0] :d 1.0}])

(defn- sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn- init-w []
  (dec (rand 2)))

(defn output [ws xs]
  (sigmoid (reduce + (map #(apply * %) (map vector ws xs)))))

(defn delta-sgd [init-ws inputs]
  (reduce
    (fn [ws {:keys [xs d]}]
      (let [y (output ws xs)
            e (- d y)
            delta (* y (- 1.0 y) e)
            dws (map #(* alpha delta %) xs)]
        (map #(apply + %) (map vector ws dws))))
    init-ws
    inputs))

(defn train [n]
  (delta-sgd
    [(init-w) (init-w) (init-w)]
    (flatten (repeat n samples))))

(defn -main [& args]
  (let [ws (train 10000)
        xs (map :xs samples)]
    (println "target ouput     : " (map :d samples))
    (println "inference output : " (map #(output ws %) xs))))

















